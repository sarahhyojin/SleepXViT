'use strict'
const log = require('util').debuglog('edf')
const Transform = require('stream').Transform
const NotFactory = new Error("Expting factory to be a function")

const STATUS_NOTSTARTED = 0
const STATUS_STARTED = 1

var RecordsStream = class extends Transform {

  /**
    Factory is function that give the file header
    and the signal headers return a RecordProcessor objects
    to be use for the translation of each data block
  */
  constructor (factory, options) {

    options = options || {}
    options.decodeStrings = true
    options.objectMode = false
    options.readableObjectMode = true
    options.writableObjectMode = false

    super(options)

    this.header = options.header || null
    this.signals = options.signals || null
    this.status = STATUS_NOTSTARTED
    this.datarecord = null

    if(!(factory instanceof Function)){
      throw NotFactory
    }

    this.factory = factory
  }

  setHeader (header) {
    this.header = header
    return this
  }

  setSignals (signals) {
    this.signals = signals
  }

  _initialize (header, signals) {

    const rprocessor = this.factory(
      header,signals
    );

    this.rprocessor = rprocessor
    this.status = STATUS_STARTED
  }

  _transform (chunk, encoding, callback) {

        // initialization
    if (STATUS_NOTSTARTED == this.status) {
      this._initialize(
        this.header, this.signals
      )
    }

    const used = this.rprocessor.Process(chunk)
    const out  = this.rprocessor.GetAll()

    this.push(out)
    //todo: findout if we need to take into accout the used value
    callback()
  }
}

module.exports = {
  RecordsStream
}
