"""Idempotently patch verl's NaiveRewardManager to dispatch compute_score concurrently
(thread pool) so the blocking 35B-judge reward calls batch instead of running serially."""
import sys
p = sys.argv[1] + "/verl/workers/reward_manager/naive.py"
s = open(p).read()
if "ThreadPoolExecutor" in s:
    print("[patch_verl] (9) concurrent reward: already present")
    sys.exit(0)
start = s.index("        for i in range(len(data)):")
end = s.index("\n        if return_dict:")
nb = '''        # gather per-sample inputs (CPU decode, fast)
        items = []
        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = dict(data_item.non_tensor_batch.get("extra_info", {}) or {})
            extra_info["num_turns"] = data_item.non_tensor_batch.get("__num_turns__", None)
            extra_info["rollout_reward_scores"] = data_item.non_tensor_batch.get("reward_scores", {})
            items.append((i, int(valid_response_length), prompt_str, response_str, ground_truth, data_source, extra_info))
        import os
        from concurrent.futures import ThreadPoolExecutor
        _n = int(os.getenv("RL_REWARD_CONCURRENCY", "256"))
        def _score(it):
            return self.compute_score(data_source=it[5], solution_str=it[3], ground_truth=it[4], extra_info=it[6])
        if _n > 1 and len(items) > 1:
            with ThreadPoolExecutor(max_workers=min(_n, len(items))) as _ex:
                scores = list(_ex.map(_score, items))
        else:
            scores = [_score(it) for it in items]
        for (i, valid_response_length, prompt_str, response_str, ground_truth, data_source, extra_info), score in zip(items, scores):
            if isinstance(score, dict):
                reward = score["score"]
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score
            reward_tensor[i, valid_response_length - 1] = reward
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str); print("[response]", response_str); print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items(): print(f"[{key}]", value)
                else: print("[score]", score)'''
open(p, "w").write(s[:start] + nb + s[end:])
print("[patch_verl] (9) concurrent reward: applied")
