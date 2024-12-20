const fs = require('fs');
const path = require('path');
const { Writable } = require('stream');

const edf = require('../module/edf-parser');
const SampleQue = require('./SampleQue');
const draw = require('./draw');
const filter = require('./filter');

module.exports = class EdfExporter {

    exportFile(inputFile, inputId, outputDir, config, jobControllerInstance) {

        /* Set input/output directories */
        const exportDir = path.join(outputDir, `${inputId}`);
        if(!fs.existsSync(exportDir)) fs.mkdirSync(exportDir);

        let sampleRates = null;
        let sampleQue = null;

        let init = true;
        let filters;

        let currentDuration = 0;
        let totalDuration = 0;
        let skipDuration = 0;
        let printedProgress = 0;

        const data2plot = (data, i) => {

            const index = i + 1;
            const imageNumber = index.toString().padStart(4, '0');
            const outputPath = path.join(exportDir, `${imageNumber}.${config.filetype}`);

            draw.exportPlot(data, config, outputPath, jobControllerInstance);

        };

        const reader = fs.createReadStream(inputFile);
        reader.on('error', (err) => {
            reject(err);
        });

        const records = edf(reader);
        records.on('error', (err) => {
            reject(err);
        });

        const processSignals = new Writable({
            objectMode: true,
            write(chunk, encoding, callback) {

                let duration = parseInt(chunk.duration, 10);

                // initialize
                if(init) {

                    sampleRates = config.signalGroups.map((group) =>
                        group.map((s) => {
                            const sampleIndex = records.signals.findIndex((e) =>
                                RegExp(s.name, 'i').test(e.Label)
                            );

                            if(sampleIndex >= 0) s.edfUnit = records.signals[sampleIndex].PhysicalDimensions; // save edf file's unit to config's each signal

                            return sampleIndex < 0
                                ? null
                                : records.signals[sampleIndex].Samples / duration;
                        })
                    );

                    filters = filter.createFilters(sampleRates, config.signalGroups);
                    sampleQue = new SampleQue(
                        config.epochLength,
                        config.signalGroups,
                        data2plot
                    );

                    totalDuration = records.header.DataRecords * duration;

                    /* Set the starting point */
                    const startTime = records.header.Start;
                    skipDuration = (60 - startTime.getSeconds()) % 30;

                    init = false;

                }

                /* Get specific signals from raw data */
                const { samples: rawSamples } = chunk;
                let sampleGroups = config.signalGroups.map((group) =>
                    group.map((s) => {
                        const sampleIndex = records.signals.findIndex((e) =>
                            RegExp(s.name, 'i').test(e.Label)
                        );
                        return sampleIndex < 0 ? [] : rawSamples[sampleIndex];
                    })
                );

                /* Apply filters */
                sampleGroups = sampleGroups.map((group, i) => {
                    let samples = group.map((s, j) =>
                        filters[i][j].lpf
                            ? s.map((ss) => filters[i][j].lpf.filter(ss))
                            : s
                    );
                    samples = samples.map((s, j) =>
                        filters[i][j].hpf
                            ? s.map((ss) => filters[i][j].hpf.filter(ss))
                            : s
                    );
                    samples = samples.map((s, j) =>
                        filters[i][j].bsf
                            ? s.map((ss) => filters[i][j].bsf.filter(ss))
                            : s
                    );
                    return samples;
                });

                if(skipDuration - duration > 0) {
                    skipDuration -= duration;
                } else {
                    /* Discard some samples if 0 < skipDuration < duration */
                    if(skipDuration >= 0) {
                        sampleGroups = sampleGroups.map((group) =>
                            group.map((s) => {
                                const len = s.length;
                                const ratio = skipDuration / duration;
                                const startIndex = Math.ceil((len - 1) * ratio);
                                return s.slice(startIndex);
                            })
                        );
                        duration -= skipDuration;
                        skipDuration = 0;
                    }

                    sampleQue.push(duration, sampleGroups);

                    /* Update progress */
                    currentDuration += duration;
                    const progress = (currentDuration / totalDuration) * 100;

                    if(progress > printedProgress) {
                        console.log(`${inputId} | Progress : ${printedProgress}%`);
                        printedProgress += 1;
                    }

                }

                callback();

            }
        });

        records
            .pipe(processSignals)
            .on('finish', () => sampleQue.flush(jobControllerInstance));

    }

}
