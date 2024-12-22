const Iir1 = require('../index')

describe('Filters', () => {
  it('Test Lowpass Filter.', function () {
    const lpf = new Iir1.LowPass();
    lpf.setup(200, 50);
    expect(lpf.filter(0)).toBeDefined();
  });

  it('Test Highpass Filter.', function () {
    const hpf = new Iir1.HighPass();
    hpf.setup(200, 0.5);
    expect(hpf.filter(0)).toBeDefined();
  });

  it('Test Bandstop Filter.', function () {
    const bsf = new Iir1.BandStop();
    bsf.setup(200, 10, 1);
    expect(bsf.filter(0)).toBeDefined();
  });
});
