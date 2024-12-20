'use strict'
const RecordsStream = require('./records.js').RecordsStream
const {DataBlocks, ParseEdfHeader} = require('./blocks.js')
const utils = require('./utils.js')
const decoder = require('./decoder.js')
const {SignalProcessor, RecordProcessor} = require('./processors.js')

exports = module.exports = createPipeChain

function createPipeChain (reader) {
  const blocks = new DataBlocks()
  const records = new RecordsStream(
        utils.StandardRecordProcessor
    )

  blocks.on('header', (header) => {
    records.setHeader(header)
  })

  blocks.on('signals', (signals) => {
    records.setSignals(signals)
  })

    // todo chain errors and close events

  return reader.pipe(blocks)
                 .pipe(records)
}


exports.DataBlocks = DataBlocks
exports.RecordsStream = RecordsStream
//utility functions
exports.utils = utils
exports.utils.ParseEdfHeader = ParseEdfHeader
exports.StandardDecoder = decoder.Decoder
exports.utils.ParseEdfHeader = ParseEdfHeader
exports.SignalProcessor = SignalProcessor
exports.RecordProcessor = RecordProcessor
