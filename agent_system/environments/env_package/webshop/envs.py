# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ray
import gym
import numpy as np

# -----------------------------------------------------------------------------
# Ray remote worker actor -----------------------------------------------------
# -----------------------------------------------------------------------------

class WebshopWorker:
    """Ray remote actor that replaces the worker function.
    Each actor hosts a *WebAgentTextEnv* instance.
    """
    
    def __init__(self, seed, env_kwargs):
        # Lazy import avoids CUDA initialisation issues
        import sys
        import os
        import glob
        import re
        import shutil
        import subprocess

        # ---- JVM pin (safety net for pyserini/pyjnius) ---------------------
        # pyserini 0.17 bundles Lucene 8 (Java 8 bytecode, class version 52.0);
        # a JVM <= Java 7 raises `UnsupportedClassVersionError: ... version 52.0`.
        # On this box `/usr/local/java` is Java 7 and the global LD_LIBRARY_PATH
        # puts its libjvm ahead of Java 11's, so dlopen picks Java 7. The run
        # script already picks a Java >= 8 and propagates it via
        # ray_init.runtime_env.env_vars; this is the in-worker safety net for
        # when that propagation doesn't land. Must run BEFORE any pyserini
        # import (which triggers `import jnius` -> dlopen libjvm).

        def _java_major(java_bin):
            try:
                r = subprocess.run([java_bin, "-version"], capture_output=True, text=True, timeout=15)
                out = r.stderr or r.stdout
            except Exception:  # pragma: no cover - best-effort
                return 0
            m = re.search(r'version "(\d+)(?:\.(\d+))?', out)
            if not m:
                return 0
            major = int(m.group(1))
            if major == 1 and m.group(2):  # legacy 1.x scheme (1.7 -> 7)
                major = int(m.group(2))
            return major

        # Scan candidates; pick the first verified Java >= 8.
        candidates = []
        for key in ("JAVA_HOME", "JDK_HOME"):
            v = os.environ.get(key)
            if v:
                candidates.append(v.rstrip("/"))
        for c in ("/usr/lib/jvm/java-11-openjdk-11.0.22.0.7-1.el7_9.x86_64",
                  "/workdir/mjdk-11.0.16", "/usr/local/jdk1.8.0_45", "/usr/local/java"):
            if c not in candidates:
                candidates.append(c)

        java_home = None
        for cand in candidates:
            jb = os.path.join(cand, "bin", "java")
            if os.path.exists(jb) and _java_major(jb) >= 8:
                java_home = cand
                break
        if java_home is None:
            java_home = (os.environ.get("JAVA_HOME") or "/usr/local/java").rstrip("/")
            print(f"[WebshopWorker] WARNING: no Java >= 8 found among {candidates}; "
                  f"using {java_home}", flush=True)

        # Pin env vars (pyjnius checks JDK_HOME before JAVA_HOME).
        os.environ["JAVA_HOME"] = java_home
        os.environ["JDK_HOME"] = java_home
        os.environ["PATH"] = os.path.join(java_home, "bin") + os.pathsep + os.environ.get("PATH", "")
        # Prepend the chosen JDK's libjvm dir so dlopen picks it up first
        # (JDK11+: lib/server; JDK8: jre/lib/*/server).
        jvm_dirs = [os.path.join(java_home, "lib", "server")]
        jvm_dirs += glob.glob(os.path.join(java_home, "jre", "lib", "*", "server"))
        new_ld = os.pathsep.join(d for d in jvm_dirs if os.path.isdir(d))
        old_ld = os.environ.get("LD_LIBRARY_PATH")
        if new_ld:
            os.environ["LD_LIBRARY_PATH"] = new_ld + (os.pathsep + old_ld if old_ld else "")
        # pyjnius starts the JVM via JNI_CreateJavaVM: ignores `_JAVA_OPTIONS`
        # (launcher-only), honours JAVA_TOOL_OPTIONS and jnius_config.add_options.
        if not os.environ.get("JAVA_TOOL_OPTIONS"):
            os.environ["JAVA_TOOL_OPTIONS"] = "-Xss256k -XX:+UseSerialGC -Xmx256m -XX:-UsePerfData"
        try:
            import jnius_config
            if not jnius_config.vm_running:
                jnius_config.add_options("-Xss256k", "-XX:+UseSerialGC", "-Xmx256m", "-XX:-UsePerfData")
        except Exception as e:  # pragma: no cover - best-effort, don't block init
            print(f"[WebshopWorker] WARNING: jnius_config preconfigure failed ({e})", flush=True)

        libjvm = "<none>"
        for seg in (os.environ.get("LD_LIBRARY_PATH") or "").split(os.pathsep):
            if seg and os.path.exists(os.path.join(seg, "libjvm.so")):
                libjvm = os.path.join(seg, "libjvm.so")
                break
        print(f"[WebshopWorker] using JAVA_HOME={java_home} "
              f"(java v{_java_major(os.path.join(java_home, 'bin', 'java')) or '?'}), "
              f"libjvm={libjvm}", flush=True)

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), 'webshop'))
        sys.path.append(project_root)
        from web_agent_site.envs import WebAgentTextEnv  # noqa: WPS433 (runtime import)

        env_kwargs['seed'] = seed
        self.env = gym.make('WebAgentTextEnv-v0', **env_kwargs)
    
    def step(self, action):
        """Execute a step in the environment"""
        obs, reward, done, info = self.env.step(action)
        info = dict(info or {})  # make a *copy* so we can mutate safely
        info['available_actions'] = self.env.get_available_actions()
        info['task_score'] = reward

        # Redefine reward. We only use rule-based reward - win for 10, lose for 0.
        if done and reward == 1.0:
            info['won'] = True
            reward = 10.0
        else:
            info['won'] = False
            reward = 0

        return obs, reward, done, info
    
    def reset(self, idx):
        """Reset the environment with given session index"""
        obs, info = self.env.reset(session=idx)
        info = dict(info or {})
        info['available_actions'] = self.env.get_available_actions()
        info['won'] = False
        return obs, info
    
    def render(self, mode_for_render):
        """Render the environment"""
        rendered = self.env.render(mode=mode_for_render)
        return rendered
    
    def get_available_actions(self):
        """Get available actions"""
        return self.env.get_available_actions()
    
    def get_goals(self):
        """Get environment goals"""
        return self.env.server.goals
    
    def close(self):
        """Close the environment"""
        self.env.close()


# -----------------------------------------------------------------------------
# Vectorised Ray environment --------------------------------------------------
# -----------------------------------------------------------------------------

class WebshopMultiProcessEnv(gym.Env):
    """A vectorised, Ray-based wrapper around *WebAgentTextEnv*.

    ``info`` dictionaries returned by :py:meth:`step` **and** :py:meth:`reset`
    automatically contain the key ``'available_actions'`` so downstream RL code
    can obtain the *legal* action set without extra IPC overhead.
    """
    def __init__(
        self,
        seed: int,
        env_num: int,
        group_n: int,
        resources_per_worker: dict,
        is_train: bool = True,
        env_kwargs: dict = None,
    ) -> None:
        super().__init__()

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()

        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        self.is_train = is_train
        if not is_train: assert group_n == 1

        self._rng = np.random.RandomState(seed)

        self._env_kwargs = env_kwargs if env_kwargs is not None else {'observation_mode': 'text', 'num_products': None}

        # -------------------------- Ray actors setup --------------------------
        env_worker = ray.remote(**resources_per_worker)(WebshopWorker)
        self._workers = []
        for i in range(self.num_processes):
            worker = env_worker.remote(seed + (i // self.group_n), self._env_kwargs)
            self._workers.append(worker)

        # Get goals from the first worker
        goals_future = self._workers[0].get_goals.remote()
        goals = ray.get(goals_future)

        # ------- original ----------#
        # if args.num is None:
        #     if split == 'test':
        #         self.goal_idxs = range(500)
        #     elif split == 'eval':
        #         self.goal_idxs = range(500, 1500)
        #     elif split == 'train':
        #         self.goal_idxs = range(1500, len(self.env.server.goals))
        # else:
        #     self.goal_idxs = range(len(self.env.server.goals))

        if not self.is_train:
            self.goal_idxs = range(500)
        else:
            self.goal_idxs = range(500, len(goals))
            
        print(self.goal_idxs)

    # ------------------------------------------------------------------
    # Base API ----------------------------------------------------------
    # ------------------------------------------------------------------

    def step(self, actions: list[str]):
        if len(actions) != self.num_processes:
            raise ValueError(
                f'Expected {self.num_processes} actions, got {len(actions)}',
            )

        # Send step commands to all workers
        futures = []
        for worker, action in zip(self._workers, actions):
            future = worker.step.remote(action)
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for obs, reward, done, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

        return obs_list, reward_list, done_list, info_list

    def reset(self):
        idx = self._rng.choice(self.goal_idxs, size=self.env_num, replace=False)
        idx = np.repeat(idx, self.group_n).tolist()

        # Send reset commands to all workers
        futures = []
        for worker, i in zip(self._workers, idx):
            future = worker.reset.remote(i)
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        obs_list, info_list = [], []
        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)

        return obs_list, info_list

    # ------------------------------------------------------------------
    # Convenience helpers ----------------------------------------------
    # ------------------------------------------------------------------

    def render(self, mode: str = 'text', env_idx: int = None):
        if env_idx is not None:
            future = self._workers[env_idx].render.remote(mode)
            return ray.get(future)

        futures = []
        for worker in self._workers:
            future = worker.render.remote(mode)
            futures.append(future)
        
        return ray.get(futures)

    # ------------------------------------------------------------------
    # Clean‑up ----------------------------------------------------------
    # ------------------------------------------------------------------

    def close(self):
        if getattr(self, '_closed', False):
            return

        # Close all workers and kill Ray actors
        close_futures = []
        for worker in self._workers:
            future = worker.close.remote()
            close_futures.append(future)
        
        # Wait for all workers to close
        ray.get(close_futures)
        
        # Kill all Ray actors
        for worker in self._workers:
            ray.kill(worker)
            
        self._closed = True

    def __del__(self):  # noqa: D401
        self.close()


# -----------------------------------------------------------------------------
# Factory helper --------------------------------------------------------------
# -----------------------------------------------------------------------------

def build_webshop_envs(
    seed: int,
    env_num: int,
    group_n: int,
    resources_per_worker: dict,
    is_train: bool = True,
    env_kwargs: dict = None,
):
    """Mirror *build_sokoban_envs* so higher‑level code can swap seamlessly."""
    return WebshopMultiProcessEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        resources_per_worker=resources_per_worker,
        is_train=is_train,
        env_kwargs=env_kwargs,
    )