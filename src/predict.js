const clarityhub = require('./FakeClarityHub')({

});

async function predict() {
    try {
        const model = await clarityhub.models.get('test-qanda');

        const questionsToAnswer = [
            'When did Chernobyl take place?',
            'When did Chernobyl meltdown?',
            'When did the Chernobyl disaster happen?',
            'Where is Chernobyl?',
            'How did the Chernobyl disaster happen?',
            'Who was to blame for the accident?',
            'How much radiation was released by the Chernobyl accident?',
            'Who was involved in cleanup efforts',
            'When was the containment stucture created?',
            'When was the sarcophagus stucture created?',
            'Who did Stephen Hawking marry?',
            'What disease did Stephen Hawking have?',
            'What illness did Stephen Hawking have?',
        ];

        await Promise.all(questionsToAnswer.map(async (question) => {
            const prediction = await model.predict(question, {
                getClusterSentences: true,
            });

            console.log('-----');
            // console.log(question, prediction.clusterSentences);
            console.log(question, prediction.sentence);
        }));
    } catch (err) {
        console.error(err);
    }
}

predict();
