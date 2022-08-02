const clustering = require('density-clustering');

const optics = new clustering.OPTICS();

module.exports = async (vectors) => {
    const clusters = optics.run(vectors, 0.7, 2);

    return clusters;
}