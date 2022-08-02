module.exports = async function similarity(t1, t2) {
    const length = (await t2.array()).length;
    // t1 must be size 1x512

    const similarityVector = [];
    const sentenceI = t1.slice([0, 0], [1]);

    for (let j = 0; j < length; j++) {
        const sentenceJ = t2.slice([j, 0], [1]);
        const sentenceITranspose = false;
        const sentenceJTransepose = true;
        const score =
            sentenceI.matMul(sentenceJ, sentenceITranspose, sentenceJTransepose)
                .dataSync();

        similarityVector.push(score[0]);
    }

    return similarityVector;
}