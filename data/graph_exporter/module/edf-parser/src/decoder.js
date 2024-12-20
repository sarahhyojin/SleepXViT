'use strict'
const InvalidDigitalRange = new Error('Invalid digital range')
const InvalidPhysicalRange = new Error('Invalid physical range')
const InvalidInput = new Error('Invalid length of byte inputs')

var Decoder = class {
  constructor (digmin, digmax, phymin, phymax) {
    const diff = digmax - digmin
    if (!diff) {
      throw InvalidDigitalRange
    }

    if(0 === (phymax - phymin)){
        throw InvalidPhysicalRange
    }

    this.m = (phymax - phymin) / diff
    this.digmax = digmax
    this.phymax = phymax
  }

  Decode (source, target) {
    const slen = source.length
    const tlen = target.length
    const pmax = this.phymax
    const dmax = this.digmax
    const m = this.m

    if (slen % 2 !== 0) {
      throw InvalidInput
    }

    var s = 0
    var t = 0
    var tmp = 0
    while (s < slen && t < tlen) {
      tmp = source.readInt16LE(s, true)

      target[t] = pmax + m * (tmp - dmax)

      s += 2
      t += 1
    }

    return s
  }
}

module.exports = {
  Decoder
}
