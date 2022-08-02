const kmeans = require('node-kmeans');

module.exports = async (vectors) => {
    const kclusters = await new Promise((resolve, reject) => {
        kmeans.clusterize(vectors, { k: 4 }, (err, res) => {
            if (err) reject(err);
            else resolve(res);
        });
    });

    console.log('--------------');
    console.log('Kmeans: ', kclusters);
    console.log('--------------');

    return kclusters;
};
