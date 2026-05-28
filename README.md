# Laboratório 10: O Pipeline Definitivo

> Partes deste laboratório foram geradas/complementadas com IA, revisadas e validadas por João Artur Veras

## Métricas de Benchmark

| Métrica | Sem Otimização | Com Otimização |
|---|---|---|
| Tempo total | 41.04s | 11.97s |
| Velocidade | 2.44 tok/s | 8.35 tok/s |
| Pico VRAM | 1189.9 MB | 1144.6 MB |
| KV Cache | ❌ | ✅ |
| Implementação de Atenção | sdpa | sdpa |

**Speedup**: 3.43x | **Redução de VRAM**: 45.3 MB (3.8%)

## Ambiente
- GPU: Tesla T4
- VRAM total: 14913 MB
- PyTorch: 2.11.0+cu128
- Modelo: TinyLlama/TinyLlama-1.1B-Chat-v1.0 (QLoRA 4-bit, NF4, double quant)
- Atenção: sdpa

## Nota sobre FlashAttention-2 vs SDPA

O enunciado original solicita FlashAttention-2, porém esta biblioteca exige GPUs
Ampere (sm_80+). A GPU utilizada neste laboratório (Tesla T4, sm_75)
não suporta FlashAttention-2.
Foi utilizado SDPA (torch.nn.functional.scaled_dot_product_attention), que oferece os
mesmos benefícios de kernel fusionado e economia de VRAM em qualquer GPU CUDA moderna.

## Parecer Técnico

**Parte A** — O colapso de VRAM descrito no problema corporativo decorre de três fatores
cumulativos: (1) o peso bruto do modelo em precisão completa (FP16), (2) a complexidade O(n²)
do Self-Attention sobre tokens de contexto RAG, e (3) o recálculo redundante dos vetores Q, K
e V a cada novo token gerado sem cache. O QLoRA em 4-bit reduziu a VRAM do modelo em ~75%
antes mesmo de processar um único token. O KV Cache eliminou o recálculo redundante e reduziu
o tempo de geração em 3.4x. O sdpa complementa ao fundir operações de softmax
e multiplicação matricial em um único kernel CUDA, usando a SRAM como buffer intermediário.

**Parte B** — O FlashAttention e o SDPA resolvem o problema de bandwidth da atenção, mas o
armazenamento do KV Cache ainda cresce linearmente com n. Com 2 milhões de tokens, um modelo
como Llama-3-8B exigiria ~32 GB apenas de KV Cache. A solução é migrar para State Space
Models (SSM), como Mamba, que substituem a atenção por uma recorrência com estado fixo:
complexidade O(1) em memória independentemente do comprimento da sequência.
