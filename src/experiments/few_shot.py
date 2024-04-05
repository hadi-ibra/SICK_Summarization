from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from overrides import overrides
from rouge import Rouge
from bert_score import score

from src.experiments.experiment import BasicExperiment


class FewShotLearning(BasicExperiment):

    def __init__(
        self, model, train_ds, eval_ds, test_ds, tokenizer, temperature, k, gen: np.random.Generator, device, logger
    ) -> None:
        super().__init__(model, train_ds, eval_ds, test_ds, tokenizer, gen, device, logger)
        self.temperature = temperature
        self.k = k
        self.dialog_max_lenght = 1024
        self.use_temperature = True if temperature > 0 else False

    @overrides
    def train(self) -> None:
        raise NotImplementedError(
            "Training for an LLM is not implemented since the required hardware capabilieties are too high"
        )

    @overrides
    def test(self, **kwargs) -> None:
        examples = self._get_examples()
        base_prompt = self._gen_examples_template(examples)

        tokens = self.tokenizer.tokenize(str(base_prompt))
        token_count = len(tokens)
        length_max = token_count + self.dialog_max_lenght

        summaries = []

        for dialog, summary_gold in self.test_ds:
            prompt = base_prompt + dialog + "SUMMARY:\n"
            inputs = self.tokenizer(prompt, return_token_type_ids=False, return_tensors="pt").to(self.device)
            generate_ids = self.model.generate(
                **inputs,
                do_sample=self.use_temperature,
                temperature=self.temperature,
                top_k=10,
                num_return_sequences=1,
                eos_token_id=self.tokenizer.eos_token_id,
                max_length=length_max,
            )
            output = self.tokenizer.batch_decode(
                generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            output_summary = output.split("SUMMARY:")[-1].strip().strip("\n")
            summaries.append((output_summary, summary_gold))

        metrics = self._compute_metrics(summaries)
        self.logger.save_results(metrics)

    @overrides
    def save(self):
        pass

    def _get_training_samples(self):
        idxes = self.gen.integers(0, len(self.train_ds), size=self.k)
        examples = []
        for idx in idxes:
            dialog, summary = self.train_ds[idx]
            examples.append((dialog, summary))
        return examples

    def _gen_examples_template(training_examples: str) -> str:
        header = "Summarize the chat dialog. Here you can find some examples:"
        tail = "Summarize the following chat dialog in one sentence.\nDIALOG:"
        examples = []
        for dialog, summary in training_examples:
            template_example = f"DIALOG: {dialog}\nSUMMARY: {summary}"
            examples.append(template_example)
        return header + " ".join(examples) + tail

    def _compute_metrics(self, summaries) -> dict:
        rouge = Rouge()
        model_summaries, gold_summaries = map(list, zip(*[s for s in summaries]))
        score_tot = rouge.get_scores(model_summaries, gold_summaries, avg=True)
        p_b, r_b, f_b = score(
            model_summaries,
            gold_summaries,
            lang="en",
            model_type="microsoft/deberta-large-mnli",
            batch_size=1,
            device=self.device,
        )
        p_bert = p_b.mean()
        r_bert = r_b.mean()
        f_bert = f_b.mean()

        score_tot["bert"] = {"r": r_bert, "p": p_bert, "f": f_bert}

        return score_tot
