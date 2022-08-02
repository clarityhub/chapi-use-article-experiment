const clustering = require('density-clustering');

const dbscan = new clustering.DBSCAN();

module.exports = async (vectors) => {
    const clusters = dbscan.run(vectors, 0.7, 2);
    console.log('--------------');
    console.log('DBSCAN: ', clusters);
    console.log('--------------');

    return clusters;
}