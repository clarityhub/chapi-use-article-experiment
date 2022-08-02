const { performance } = require('perf_hooks');

const tf = require('@tensorflow/tfjs-node');
const Cacheman = require('cacheman');
const nlp = require('compromise');
const sentiment = require('node-sentiment');

const UniversalSentenceEncoder = require('./use');
const similarity = require('./similarity');

const use = new UniversalSentenceEncoder();

const cache = new Cacheman('ch-models', {
    engine: 'file',
    ttl: '1hr',
});

function indexOfMax(arr) {
    if (arr.length === 0) {
        return -1;
    }

    var max = arr[0];
    var maxIndex = 0;

    for (var i = 1; i < arr.length; i++) {
        if (arr[i] > max) {
            maxIndex = i;
            max = arr[i];
        }
    }

    return maxIndex;
}

class UtteranceData {
    constructor(id, utterance, embedding, sentiment, language, topics) {
        this.id = id;
        this.utterance = utterance;
        this.embedding = embedding;
        this.sentiment = sentiment;
        this.language = language;
        this.topics = topics;
    }
}

async function getEmbeddings(use, utterances) {
    const embeddings = await use.embed(utterances);
    const embeddingVectors = await embeddings.array();

    return embeddingVectors;
} 

async function getSentimentAndLanguage(utterance) {
    const { comparative, language } = sentiment(utterance);
    return [comparative, language];
}

async function getTopics(utterance) {
    const doc = nlp(utterance);
    
    return doc.topics().out('array');
}

class Model {
    constructor(use, name) {
        this.name = name;
        this.use = use;

        this.utteranceData = [];
    }

    async train(utterances) {
        performance.mark('Embed Total Start');

        const embeddings = await getEmbeddings(this.use, utterances);

        performance.mark('Embed Total End');
        performance.measure('Embed Total', 'Embed Total Start', 'Embed Total End');

        performance.mark('Utterance Data Total Start');

        const utteranceData = await Promise.all(utterances.map(async (utterance, i) => {
            const [embedding, [sentiment, language], topics] = await Promise.all([
                embeddings[i],
                getSentimentAndLanguage(utterance),
                getTopics(utterance),
            ]);

            return new UtteranceData(
                i,
                utterance,
                embedding,
                sentiment,
                language,
                topics
            );
        }));

        performance.mark('Utterance Data Total End');
        performance.measure('Utterance Data Total', 'Utterance Data Total Start', 'Utterance Data Total End');

        // XXX Create a similarity matrix 
        this.utteranceData = [...this.utteranceData, ...utteranceData];

        // const embeddings = await this.model.embed(sentences);
        // const embeddingVectors = await embeddings.array();
        // const clusters = await optics(embeddingVectors);

        // const mappings = mapClustersToSentences(clusters, sentences);
        // const modelDataSentences = zip(sentences, embeddingVectors);

        // const modelData = {
        //     clusters: clusters.map((c, i) => new Cluster(i, c, mappings[i])),
        //     sentences: modelDataSentences.map(([sentence, embedding], i) => {
        //         return new Sentence(i, sentence, embedding);
        //     }),
        // }

        // // XXX modelData should not be a destructive action
        // // each training session should be additive.
        // this.modelData = modelData;

        // Save model data to disk
        await new Promise((resolve, reject) => {
            cache.set(this.name, this.utteranceData, (err) => {
                if (err) reject();
                resolve();
            });
        });
    }

    /**
     * Given an utterance, compare it to the entire model
     *
     * @param {String} utterance 
     */
    async predict(utterance) {
        const newEmbeddings = await this.use.embed([utterance]);

        const embeddingVectors = this.utteranceData.map(utterance => {
            return utterance.embedding;
        });

        const similarityVector = await similarity(
            newEmbeddings,
            tf.tensor(embeddingVectors),
        );

        const index = indexOfMax(similarityVector);

        let bestFitSentence = this.utteranceData.find(u => u.id === index).utterance;
    
        return {
            sentence: bestFitSentence,
        };
    }
}

Model.load = async function (use, name) {
    const utteranceData = await new Promise((resolve, reject) => {
        cache.get(name, (err, value) => {
            if (err) reject();
            resolve(value);
        });
    });

    if (!utteranceData) {
        throw new Error(`Model Data does not exist for ${name}`);
    }

    const model = new Model(use, name);
    model.utteranceData = utteranceData;

    return model;
};

module.exports = () => ({
    models: {
        create: async function (name) {
            await use.load();

            return new Model(use, name);
        },

        get: async function (name) {
            await use.load();

            return Model.load(use, name);
        }
    }
});
