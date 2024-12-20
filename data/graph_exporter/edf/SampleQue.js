/* Accumulate samples for certain duration
 * and flush them into graph drawer */

module.exports = class SampleQue {

    constructor(maxDuration, signalConfig, callback) {
        this.maxDuration = maxDuration;
        this.signalConfig = signalConfig;
        this.callback = callback;

        this.sampleGroups = null;
        this.totalDuration = 0;
        this.count = 0;
    }

    push(duration, sampleGroups) {

        this.totalDuration += duration;
        if (!this.sampleGroups) {
            this.sampleGroups = sampleGroups;
        } else {
            this.sampleGroups = this.sampleGroups.map((group, i) => {
                const newSamples = sampleGroups[i];
                return group.map((s, j) => s.concat(newSamples[j]));
            });
        }

        /* Flush the data into the callback */
        if (this.totalDuration >= this.maxDuration) {
            const dataDuration = this.maxDuration;
            const ratio = dataDuration / this.totalDuration;

            const dataGroups = [];
            const remainingGroups = [];
            for (let i = 0; i < this.sampleGroups.length; i += 1) {
                const samples = this.sampleGroups[i];
                const dataSamples = [];
                const remaining = [];
                for (let j = 0; j < samples.length; j += 1) {
                    const s = samples[j];
                    const indexToSplit = Math.floor(s.length * ratio);
                    dataSamples.push(s.slice(0, indexToSplit + 1));
                    remaining.push(s.slice(indexToSplit));
                }
                dataGroups.push(dataSamples);
                remainingGroups.push(remaining);
            }

            const data = {
                signalConfig: this.signalConfig,
                groups: dataGroups,
                duration: dataDuration,
                maxDuration: this.maxDuration,
            };

            this.callback(data, this.count);

            this.sampleGroups = remainingGroups;
            this.totalDuration -= dataDuration;
            this.count += 1;
        }

    }

    flush(jobControllerInstance) {

        // if (this.sampleGroups && this.totalDuration > 0) {

            const paddedGroups = this.sampleGroups.map((group) =>
                group.map((sample) => [...sample, 0])
            );
            const data = {
                signalConfig: this.signalConfig,
                groups: paddedGroups,
                duration: this.totalDuration,
                maxDuration: this.maxDuration,
            };

            this.callback(data, this.count);

            jobControllerInstance.finish();

            this.sampleGroups = null;

        // }

    }

}
