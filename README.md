# USE Article Experiment

Experiment to test how well USE can work to do Question and Answers from
articles.

## 🚀 Quick Start

```sh
npm i
npm run train
npm run predict
```

## Idea

Can we use the Universal Sentence Encoder to find answers to questions?

## Hypothesis

If we encode both paragraphs and questions using USE, then we can provide better answers to questions by comparing the embeddings than by using simple text comparison like ElasticSearch.

## Method

1. Pull down data for articles from Wikipedia
2. Run the paragraphs through USE
3. Save those embeddings
4. Have a set of questions to ask those articles
5. Run those questions through USE
6. Find the most similar vector to the question embedding from the article embeddings
7. Use that paragraph as the answer to the question.

Still TODO:

1. Set up Elasticsearch
2. Put each paragraph as an entry into Elasticsearch
3. Do a text search using the question against Elasticsearch

1. Compare results
    1. Go through both articles and rank the possible paragraphs that can answer the questions. Give them scores from 10 to 1 (if there are 10 answers).
    2. Rank the answers given by the two systems using the scores.
    3. Compare the overall rankings of both systems.


