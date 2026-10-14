"""LLM option scorer + Appendix-B prompt. Logits = sequence log-probs of each option string, log-softmaxed over the 3 options."""
import numpy as np
import torch

TEMPLATE = """You are a classifier.

Which of the following three professions best matches the person described in the biography below? Choose exactly one:
- {op1}
- {target}
- {op2}

Output only the profession name.

Biography:
\"\"\"
{bio}
\"\"\"
{context}
Answer:"""


def build_prompt(bio, target, d1, d2, passages=None, preamble=None, ctx_words=60):
    """passages: list of (text, gender) from the reranked slate S_K (truncated, annotated with metadata gender)."""
    ctx = ""
    if passages:
        lines = ["Context documents:"]
        for t, g in passages:
            lines.append("{text: %s, gender: %s}" % (" ".join(str(t).split()[:ctx_words]), "female" if g == 1 else "male"))
        ctx = "\n".join(lines) + "\n\n"
    p = TEMPLATE.format(op1=d1.replace("_", " "), target=target.replace("_", " "), op2=d2.replace("_", " "),
                        bio=bio, context=ctx)
    return (preamble + "\n\n" + p) if preamble else p


class OptionScorer:
    def __init__(self, model_name, seq2seq=False, dtype=torch.bfloat16, device=None, max_len=2048):
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        cls = AutoModelForSeq2SeqLM if seq2seq else AutoModelForCausalLM
        self.model = cls.from_pretrained(model_name, torch_dtype=dtype).to(self.dev).eval()
        self.seq2seq, self.max_len = seq2seq, max_len

    @torch.no_grad()
    def seq_logprobs(self, prompt, options):
        """log P(option | prompt) summed over option tokens, for each option."""
        n = len(options)
        if self.seq2seq:
            enc = self.tok([prompt] * n, return_tensors="pt", truncation=True, max_length=self.max_len,
                           padding=True).to(self.dev)
            lab = self.tok(options, return_tensors="pt", padding=True).input_ids.to(self.dev)
            mask = lab != self.tok.pad_token_id
            logits = self.model(**enc, labels=lab.masked_fill(~mask, -100)).logits.float().log_softmax(-1)
            lp = logits.gather(-1, lab.unsqueeze(-1)).squeeze(-1)
            return (lp * mask).sum(1).cpu().numpy()
        p_ids = self.tok(prompt, add_special_tokens=True).input_ids
        o_ids = [self.tok(" " + o, add_special_tokens=False).input_ids for o in options]
        room = self.max_len - max(map(len, o_ids))
        p_ids = p_ids[:1] + p_ids[1:][-(room - 1):] if len(p_ids) > room else p_ids  # keep BOS, drop oldest context
        L = len(p_ids) + max(map(len, o_ids))
        ids = torch.full((n, L), self.tok.pad_token_id); att = torch.zeros((n, L), dtype=torch.long)
        for i, o in enumerate(o_ids):
            s = p_ids + o
            ids[i, :len(s)] = torch.tensor(s); att[i, :len(s)] = 1
        ids, att = ids.to(self.dev), att.to(self.dev)
        lp = self.model(input_ids=ids, attention_mask=att).logits.float().log_softmax(-1)
        out = []
        for i, o in enumerate(o_ids):
            pos = torch.arange(len(p_ids) - 1, len(p_ids) - 1 + len(o), device=self.dev)
            out.append(lp[i, pos, torch.tensor(o, device=self.dev)].sum().item())
        return np.array(out)

    def option_logits(self, prompt, target, d1, d2):
        """Returns log-softmax over [target, d1, d2] (target first)."""
        lp = self.seq_logprobs(prompt, [target.replace("_", " "), d1.replace("_", " "), d2.replace("_", " ")])
        lp = lp - lp.max()
        return lp - np.log(np.exp(lp).sum())

    def yes_no(self, prompt):
        """logit(Yes) - logit(No) for the filter baselines."""
        lp = self.seq_logprobs(prompt, ["Yes", "No"])
        return float(lp[0] - lp[1])
