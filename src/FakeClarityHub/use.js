const { performance } = require('perf_hooks');

const path = require('path');
const tf = require('@tensorflow/tfjs-node');
const { Tokenizer } = require('@tensorflow-models/universal-sentence-encoder');

async function loadVocabulary() {
    return require('./data/tensor-vocab.json');
}

const handler = tf.io.fileSystem(path.resolve(__dirname + '/data/tensor-model.json'));

module.exports = class UniversalSentenceEncoder {
    async loadModel() {
        return tf.loadGraphModel(handler);
    }

    async load() {
        const [model, vocabulary] =
            await Promise.all([this.loadModel(), loadVocabulary()]);

        this.model = model;
        this.tokenizer = new Tokenizer(vocabulary);
    }

    /**
     *
     * Returns a 2D Tensor of shape [input.length, 512] that contains the
     * Universal Sentence Encoder embeddings for each input.
     *
     * @param inputs A string or an array of strings to embed.
     */
    async embed(inputs) {
        if (typeof inputs === 'string') {
            inputs = [inputs];
        }

        performance.mark('Embed Tokenizer Start');

        // XXX This takes the most amount of time
        const encodings = inputs.map(d => this.tokenizer.encode(d));

        performance.mark('Embed Tokenizer End');
        performance.measure('Embed Tokenizer', 'Embed Tokenizer Start', 'Embed Tokenizer End');

        const indicesArr =
            encodings.map((arr, i) => arr.map((d, index) => [i, index]));

        let flattenedIndicesArr = [];
        for (let i = 0; i < indicesArr.length; i++) {
            flattenedIndicesArr =
                flattenedIndicesArr.concat(indicesArr[i]);
        }

        const indices = tf.tensor2d(
            flattenedIndicesArr, [flattenedIndicesArr.length, 2], 'int32');
        const values = tf.tensor1d(tf.util.flatten(encodings), 'int32');

        performance.mark('Embed Execute Start');
        
        const embeddings = await this.model.executeAsync({ indices, values });

        performance.mark('Embed Execute End');
        performance.measure('Embed Execute', 'Embed Execute Start', 'Embed Execute End');

        indices.dispose();
        values.dispose();

        return embeddings;
    }
}
