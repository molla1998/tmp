This project can easily generate **40–60 interview questions**. I'd group them as follows:

---

# 1. High-Level Architecture (Most Common)

### Q1. Explain your end-to-end system.

Expected:

```text
Product Catalog
↓
RAG + LLM
↓
Synthetic Dataset
↓
LoRA Fine-tuning Gemma-3
↓
Query Understanding Agent
↓
NER + Intent + Tab
↓
NSFW Guardrails
↓
Structured Output
↓
Search API
```

---

### Q2. Why use an LLM instead of DistilBERT?

Expected:

* multilingual support
* single model for multiple tasks
* structured output generation
* easier extensibility

---

### Q3. Why Gemma-3?

Expected:

* open weights
* strong multilingual capabilities
* efficient fine-tuning
* deployable locally

---

### Q4. Why LoRA instead of full fine-tuning?

Expected:

* fewer trainable params
* lower GPU memory
* faster training
* cheaper experimentation

---

# 2. LoRA Deep Dive (VERY COMMON)

### Q5. What is LoRA?

Expected:

```text
freeze base model
train low-rank matrices
```

---

### Q6. Explain LoRA mathematically.

Expected:

W'=W+BA

where:

* W frozen
* A,B trainable

---

### Q7. What is rank r in LoRA?

Expected:

* bottleneck dimension
* controls trainable parameters

---

### Q8. Why does LoRA work?

Expected:

* task updates are low-rank
* don't need full parameter updates

---

# 3. Multi-Task Learning

### Q9. How are NER, Intent and Tab Prediction done together?

Expected:

* single LLM
* structured output

---

### Q10. Why multitask learning?

Expected:

* shared knowledge
* fewer models
* better generalization

---

### Q11. How were losses combined?

Expected:

L=\lambda_1L_{NER}+\lambda_2L_{Intent}+\lambda_3L_{Tab}

---

### Q12. How did you choose λ values?

Expected:

* empirical tuning
* monitor task performance

---

### Q13. What is Negative Transfer?

Expected:

* one task hurts another task

---

# 4. NER Questions

### Q14. What tagging scheme used?

Expected:

* BIO

---

### Q15. What happens when a word splits into subwords?

Expected:

* propagate labels
  or
* ignore subword loss

---

### Q16. Why use F1 for NER?

Expected:

* class imbalance
* entity correctness

---

### Q17. Entity-level vs Token-level F1?

VERY SENIOR

Expected:

* entity-level stricter

---

# 5. Intent Classification

### Q18. Why not rule-based intent detection?

Expected:

* scale
* language variability

---

### Q19. What intents did you support?

Examples:

* product search
* support
* accessory
* warranty

---

### Q20. How do you handle ambiguous queries?

Example:

```text
iphone
```

Expected:

* context
* confidence score
* fallback routing

---

# 6. Synthetic Data Generation

MOST LIKELY

---

### Q21. Why synthetic data?

Expected:

* annotation expensive
* multilingual coverage

---

### Q22. How was data generated?

Expected:

```text
Catalog
↓
Retriever
↓
Prompt
↓
LLM
↓
Query + Labels
```

---

### Q23. Did synthetic data create bias?

Expected:

* yes possible
* repetitive patterns
* unrealistic distribution

Mitigation:

* diversity prompts
* mix real queries

---

### Q24. How did you ensure data quality?

Expected:

* schema validation
* confidence filtering
* deduplication
* manual audits

---

### Q25. Why RAG?

Expected:

* catalog grounding
* factual consistency

---

# 7. RAG Questions

### Q26. Explain RAG.

Expected:

```text
Retrieve
+
Generate
```

---

### Q27. Why embeddings?

Expected:

* semantic retrieval

---

### Q28. Why not put entire catalog in prompt?

Expected:

* context limits
* hallucinations
* cost

---

### Q29. What embedding model did you use?

Be ready.

---

### Q30. Cosine similarity formula?

\cos(\theta)=\frac{A\cdot B}{||A||||B||}

---

# 8. Guardrails / NSFW

### Q31. How did NSFW filtering work?

Expected:

* rule-based
* classifier
* moderation model

---

### Q32. Before or after LLM?

Expected:

* before LLM

---

### Q33. Why before?

Expected:

* safety
* latency
* cost

---

# 9. Search Concepts

VERY IMPORTANT

### Q34. How does query understanding improve CTR?

Expected:

* better intent understanding
* better retrieval

---

### Q35. What caused the 31% CTR improvement?

Expected:

* A/B testing
* improved routing
* better entity extraction

---

### Q36. What metrics were tracked?

Expected:

* CTR
* NDCG
* MRR
* Precision
* Recall

---

### Q37. Explain NDCG.

---

### Q38. Explain MRR.

---

### Q39. Explain MAP.

---

### Q40. Why F1 instead of accuracy?

---

# 10. Quantization / Deployment

VERY COMMON

### Q41. Why INT8 Quantization?

Expected:

* smaller model
* faster inference

---

### Q42. Why 4× reduction?

Expected:

```text
FP32 → INT8
32 bits → 8 bits
```

---

### Q43. Accuracy drop after quantization?

Expected:

* small degradation

---

### Q44. Dynamic vs Static Quantization?

Senior question.

---

### Q45. Why ONNX?

Expected:

* optimized inference
* graph fusion

---

# 11. Multilingual NLP

### Q46. Why train one multilingual model?

Expected:

* shared representations
* lower maintenance

---

### Q47. Challenges with Korean and Arabic?

Expected:

* tokenization
* morphology
* script differences

---

### Q48. How was language imbalance handled?

Expected:

* sampling
* weighting

---

# 12. Senior-Level Questions

### Q49. What was the biggest bottleneck?

---

### Q50. If accuracy dropped in Arabic, what would you do?

---

### Q51. How would you scale to 50 languages?

---

### Q52. How would you continuously improve the model?

Expected:

* feedback loop
* active learning
* retraining

---

### Q53. What would you do if CTR improved but NER F1 dropped?

VERY SENIOR

Expected:

```text
business metric > offline metric
```

Need root-cause analysis.

---

### Q54. How would you redesign this with an Agentic workflow?

Expected:

```text
NSFW Tool
↓
NER Tool
↓
Intent Tool
↓
Search Tool
↓
LLM Orchestrator
```

---

If I were interviewing for an ML/NLP/Search role, the **top 15 I'd expect almost certainly** are:

1. Explain architecture
2. Why Gemma-3
3. Why LoRA
4. LoRA math
5. How multitask learning works
6. Loss combination
7. Negative transfer
8. Why synthetic data
9. RAG architecture
10. Data quality checks
11. Why F1
12. Explain CTR improvement
13. Explain NDCG
14. INT8 quantization
15. Multilingual challenges

These cover about 80% of likely discussion around this project.
