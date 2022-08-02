module.exports = async function tensordot(embeddings, length) {
    const dotMatrix = [];

    for (let i = 0; i < length; i++) {
        dotMatrix.push([]);
    }

    for (let i = 0; i < length; i++) {
        for (let j = i; j < length; j++) {
            const sentenceI = embeddings.slice([i, 0], [1]);
            const sentenceJ = embeddings.slice([j, 0], [1]);
            const sentenceITranspose = false;
            const sentenceJTransepose = true;
            const score =
                sentenceI.matMul(sentenceJ, sentenceITranspose, sentenceJTransepose)
                    .dataSync();

            dotMatrix[i][j] = score[0];
            dotMatrix[j][i] = score[0];
        }
    }

    return dotMatrix;
}