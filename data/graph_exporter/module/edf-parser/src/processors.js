'use strict'
// signal processors are in charge of process the set of raw bytes from the data block
// and convert them to floats
// they follow a simple interface:
// int function Sample() that return the amount of samples it returns
// int function Process(chunk,out)-> convert the chunks into float and copy them into array out

const IncompleteSignalData = new Error('Signal data incomplete')
const IncompleteDecode = new Error('Incomplete data decoding')
const IncompleteRecord = new Error('Incomplete data record')
const OutofRange = new Error('Signal index out of range')
const NoRecord = new Error('No record has been read')

var SignalProcessor = class {
  constructor (header, decoder) {
        // signal header
    this.header = header
    this.decoder = decoder
  }

  Samples () {
    return this.header.Samples
  }

  Process (chunk, out) {
    const toProcess = this.Samples() * 2

    if (chunk.length < toProcess) {
      throw IncompleteSignalData
    }

    const decoded = this.decoder.Decode(
            chunk.slice(0, toProcess), out
        )

    if (decoded !== toProcess) {
      throw IncompleteDecode
    }

    return decoded
  }
}

// record processors are in charge of processing and entire data record
// they act as coordinatior between the raw data and each signal processor in the data record
// they follow the below insterface:
// Process(chunk) were chunk  is a full data record in a buffer returns the amount of bytes processed from the chunk
// StartTime returns the adjusted StartTime base on duration of the given data record
// GetAll() returns and object with all the processed record

var RecordProcessor = class {
  constructor (startTime, duration, processors) {
    var samples = []

    processors.forEach(function (p) {
      samples.push(
        new Array(p.Samples()).fill(0.0)
      )
    })

    this.processors = processors
    this.samples = samples
    this.count = 0
    this.start = startTime
    this.duration = duration
  }

  Process (chunk) {
    var samples = this.samples
    const processors = this.processors
    const length = chunk.length
    var current = 0

    for (let i = 0; i < this.processors.length; i++) {
      current += processors[i].Process(
        chunk.slice(current), samples[i]
      )

      if(current > length){
        throw IncompleteRecord
      }
    }

    this.count++
    return current
  }

  StartTime () {

    if(0 === this.count){
        throw NoRecord;
    }

    return new Date(
        //convert to milliseconds and back
        this.start.getTime() + (this.count-1)*(1000*this.duration)
    );
  }

  Get (index) {

    if(0 === this.count){
        throw NoRecord;
    }

    if (index < 0 && index >= this.samples.length) {
      throw OutofRange
    }

    return this.samples[index]
  }

  GetAll () {
    if(0 === this.count){
        throw NoRecord;
    }

    return {
      count: this.count,
      start: this.StartTime(),
      duration: this.duration,
      samples: this.samples
    }
  }
}

module.exports = {
  SignalProcessor,
  RecordProcessor
}
