const { PerformanceObserver, performance } = require('perf_hooks');

const wtf = require('wtf_wikipedia');
const Cacheman = require('cacheman');

const clarityhub = require('./FakeClarityHub')({
    apiKeyId: process.env.API_KEY_ID,
    apiKeySecret: process.env.API_KEY_SECRET,
});

const cache = new Cacheman('app', {
    engine: 'file',
    ttl: '1hr',
});

async function getWikiData(request) {
    // Read from Cache
    const cacheValue = await new Promise((resolve, reject) => {
        cache.get(request, function (err, value) {
            if (err) {
                reject(err);
            }

            resolve(value);
        });
    });

    if (cacheValue) {
        console.log('✅ cache hit');

        return cacheValue;
    }

    const doc = await wtf.fetch(request, 'en');
    const paragraphs = doc.paragraphs();

    const plaintext = paragraphs.slice(0, 120).map((p) => {
        const parts = p.sentences();

        if (parts.length >= 6) {
            // Split up
            return [
                parts.slice(0, parts.length / 2).map(p => p.text()).join(' '),
                parts.slice(parts.length / 2).map(p => p.text()).join(' '),
            ];
        }

        return [parts.map(p => p.text()).join(' ')];
    }).reduce((acc, p) => {
        return [...acc, ...p];
    }, []);

    // Set Cache data
    await new Promise((resolve, reject) => {
        cache.set(request, plaintext, function (error) {
            if  (error) {
                reject(error);
            }

            resolve();
        });
    })

    return plaintext;
}

async function train() {
    try {
        const result = await getWikiData('Chernobyl Disaster');
        const result2 = await getWikiData('Stephen Hawking');

        const model = await clarityhub.models.create('test-qanda');

        console.log('Training on');
        console.log(` - Utterances: + ${result.length}`);
        console.log(` - Utterances: + ${result2.length}`);
        
        await monitorPerformance(async function () {
            await model.train([...result, ...result2]);
        });
    } catch (err) {
        console.log(err);
    }
}

async function monitorPerformance(callback) {
    performance.mark('Train Total Start');

    await callback();

    performance.mark('Train Total End');
    performance.measure('Train Total', 'Train Total Start', 'Train Total End');

    const used = process.memoryUsage();
    for (let key in used) {
        console.log(`${key} ${Math.round(used[key] / 1024 / 1024 * 100) / 100} MB`);
    }
}

const obs = new PerformanceObserver((items) => {
    console.log(`${items.getEntries()[0].name}: ${items.getEntries()[0].duration} milliseconds`);
    performance.clearMarks();
});

obs.observe({ entryTypes: ['measure'] });

train();
