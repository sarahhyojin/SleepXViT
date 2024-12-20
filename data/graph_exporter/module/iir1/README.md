# IIR1.JS - NodeJS binding of realtime C++ filter library

## How To Use

### Installation

```bash
yarn add ssh://git@gitlab.com:kongdole/iir1-js.git
```

### Creating and applying a filter

```javascript
const iir = require('iir1');

const filter = iir.HighPass();
const sampleFreq = 200; // Sampling Frequency = 200Hz
const cutoffFreq = 10;  // Cutoff Frequency = 10Hz
filter.setup(sampleFreq, cutoffFreq);

const data = [3.5, 10, 2.7, -8.3, 6.9] // Sample data
const filteredData = data.map((d) => filter.filter(d));
```

### Supported filters

For now, only butterworth filter is supported.

1. Lowpass Filter

```javascript
const lpf = iir.LowPass();
lpf.setup(sampleFreqency, cutoffFrequency);
```

2. Highpass Filter

```javascript
const hpf = iir.HighPass();
hpf.setup(sampleFreqency, cutoffFrequency);
```

3. BandStop Filter

```javascript
const bsf = iir.BandStop();
bsf.setup(sampleFreqency, stopFrequency, widthFrequency);
```
