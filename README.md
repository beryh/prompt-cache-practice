# Prompt cache 활용 실습
- Amazon Bedrock의 Prompt cache를 활용하는 실습입니다.
- Lab 1은 Document RAG, Lab2는 Multi Turn 대화를 주제로 합니다.

## Lab 1. Document RAG
- 주어진 문서(예제는 [`로미오와 줄리엣`](./documents/romeo_and_juliet_korean.pdf) 입니다.)의 내용을 바탕으로 질문을 합니다.
- Prompt Cache를 사용하지 않는 답변과 Prompt Cache를 사용하는 답변을 통해, 처리 시간과 소모 토큰을 비교합니다.
- 예제에서는 1막의 내용만 활용하지만, 전체 문서를 대상으로 Prompt Cache를 적용한다면 더 높은 효율을 볼 수 있습니다.

## Lab 2. Multi-turn Chatting
- 주어진 문서(예제는 [`Effectively use prompt caching on Amazon Bedrock`](./documents/prompt_caching_article.md))의 내용을 바탕으로 연속된 형태의 질문을 합니다.
- Cache Point의 위치에 유의하며 실습을 수행합니다.
- 앞선 Conversation의 문답을 재사용합니다.

## Appendix
- [`Converse API`](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html)
- [`Effectively use prompt caching on Amazon Bedrock`](https://aws.amazon.com/blogs/machine-learning/effectively-use-prompt-caching-on-amazon-bedrock/)
