'use strict'
const processors = require('./processors.js')
const decoders = require('./decoder.js')

function StandardSignalProcessors (headers) {
  var list = headers.map((h) => {
    const decoder = new decoders.Decoder(
      h.DigitalMin, h.DigitalMax, h.PhysicalMin, h.PhysicalMax
    )

    return new processors.SignalProcessor(
      h, decoder
    )
  })

  return list
}

function StandardRecordProcessor (header, signals) {
  const list = StandardSignalProcessors(signals)

  return new processors.RecordProcessor(
    header.Start, header.Duration, list
  )
}

module.exports = {
  StandardSignalProcessors,
  StandardRecordProcessor
}
