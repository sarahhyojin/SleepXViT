const Iir1 = require('../module/iir1');

const createFilters = (samplingRates, signalConfig) => {

    const filters = signalConfig.map((group, i) =>
        group.map((s, j) => {
            let lpf;
            let hpf;
            let bsf;

            if (!samplingRates[i][j]) {
                return { lpf, hpf, bsf };
            }
            ////////////////////////////////////////////////////////
            //Ensure low-pass filter cutoff frequency is valid
            if (s.lp && s.lp >= samplingRates[i][j]) {
               s.lp = samplingRates[i][j] - 1;
               console.warn('Cutoff frequency for low-pass filter adjusted to be below the sampling frequency.');
            }

            //Ensure high-pass filter cutoff frequency is valid
            if (s.hp && s.hp >= samplingRates[i][j]) {
               s.hp = samplingRates[i][j] - 1;
               console.warn('Cutoff frequency for high-pass filter adjusted to be below the sampling frequency.');
            }

            //Ensure band-stop filter cutoff frequency is valid
            if (s.bs && (s.bs >= samplingRates[i][j] || s.bs2 >= samplingRates[i][j])) {
                //Adjust either bs or bs2 (you may need additional logic based on your requirements)
               s.bs = Math.min(s.bs, samplingRates[i][j] - 1);
               console.warn('Cutoff frequency for band-stop filter adjusted to be below the sampling frequency.');
            }

            //Ensure low-pass filter cutoff frequency is below the Nyquist frequency
            if (s.lp && s.lp >= samplingRates[i][j] / 2) {
               s.lp = samplingRates[i][j] / 2 - 1;
               console.warn('Cutoff frequency for low-pass filter adjusted to be below the Nyquist frequency.');
            }

            //Ensure high-pass filter cutoff frequency is below the Nyquist frequency
            if (s.hp && s.hp >= samplingRates[i][j] / 2) {
               s.hp = samplingRates[i][j] / 2 - 1;
               console.warn('Cutoff frequency for high-pass filter adjusted to be below the Nyquist frequency.');
            }

            //Ensure band-stop filter cutoff frequency is below the Nyquist frequency
            if (s.bs && (s.bs >= samplingRates[i][j] / 2 || s.bs2 >= samplingRates[i][j] / 2)) {
                //Adjust either bs or bs2 (you may need additional logic based on your requirements)
               s.bs = Math.min(s.bs, samplingRates[i][j] / 2 - 1);
               console.warn('Cutoff frequency for band-stop filter adjusted to be below the Nyquist frequency.');
            }
            /////////////////////////

            if (s.lp) {
                lpf = new Iir1.LowPass();
                lpf.setup(samplingRates[i][j], s.lp);
            }

            if (s.hp) {
                hpf = new Iir1.HighPass();
                hpf.setup(samplingRates[i][j], s.hp);
            }

            if (s.bs) {
                bsf = new Iir1.BandStop();
                bsf.setup(samplingRates[i][j], s.bs, 1);
            }

            return { lpf, hpf, bsf };
        })
    );

    return filters;

}

module.exports = {
    createFilters: createFilters
};
