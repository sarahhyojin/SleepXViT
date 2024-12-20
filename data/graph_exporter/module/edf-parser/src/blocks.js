'use strict'
const Transform = require('stream').Transform
const log = require('util').debuglog('edf')

const ENCODING = 'ascii'
const DATE_SEP = '.'
const HEADER_LENGTH = 256
const HEADER_STATUS = 0
const SIGNAL_STATUS = 1
const DATA_STATUS = 2

// errors
const InvalidStart = new Error('Invalid start date/time field')
const InvalidHeaderBuffer = new Error('Edf header encoded can no be less than 256')

const signal_header_fields = [
  {
    name: 'Label',
    length: 16
  },
  {
    name: 'Transducer',
    length: 80
  },
  {
    name: 'PhysicalDimensions',
    length: 8
  },
  {
    name: 'PhysicalMin',
    length: 8,
    cast: parseFloat
  },
  {
    name: 'PhysicalMax',
    length: 8,
    cast: parseFloat
  },
  {
    name: 'DigitalMin',
    length: 8,
    cast: parseInt
  },
  {
    name: 'DigitalMax',
    length: 8,
    cast: parseInt
  },
  {
    name: 'Prefiltering',
    length: 80
  },
  {
    name: 'Samples',
    length: 8,
    cast: parseInt
  },
  {
    name: 'Reserved',
    length: 32
  }
]

function TransformState () {
  this.stage = HEADER_STATUS
  this.buffer = Buffer.allocUnsafe(HEADER_LENGTH)
  this.waterMark = 0
  this.headers = null
  this.signals = null
  this.block = null
  this.error = null
}

var DataBlocks = class extends Transform {
  constructor (options) {
    options = options || {}
    options.decodeStrings = true
    options.readableObjectMode = false
    options.writableObjectMode = false
    super(options)

    this.headerParser = options.headerParser || ParseEdfHeader
    this.state = new TransformState()
  }

  async _transform (chunk, encoding, callback) {
    if (HEADER_STATUS === this.state.stage) {
      const used = read_header(this.state, chunk, this.headerParser)
      if (used && !this.state.error) {
        // we are assuming header.Signals is always > 0
        this.state.stage = SIGNAL_STATUS
        chunk = chunk.slice(used)
        this.emit('header', this.state.header)
      }
    }

    if (SIGNAL_STATUS === this.state.stage) {
      const used = read_signal_headers(this.state, chunk)
      if (used && !this.state.error) {
        this.state.stage = DATA_STATUS
        chunk = chunk.slice(used)
        this.emit('signals', this.state.signals)
      }
    }

    if (DATA_STATUS === this.state.stage) {
      while (chunk.length > 0) {
        const {used, parsed} = read_data_block(this.state, chunk)
        chunk = chunk.slice(used)
        if (parsed) {
          await this.push(this.state.block)
        }
      }
    }

    callback(this.state.error)
  }
}

function read_header (status, data, parser) {
  const to_copy = HEADER_LENGTH - status.waterMark

  const copied = data.copy(
        status.buffer, status.waterMark, 0, Math.min(to_copy, data.length)
    )

  status.waterMark += copied
  if (HEADER_LENGTH !== status.waterMark) {
    return false
  }

  var header = {}
  const result = parser(status.buffer, header)
  if (result !== true) {
    status.error = result
    return false
  }

  status.header = header
  status.waterMark = 0

  return copied
}

function ParseEdfHeader (buffer, header) {

  if(buffer.length < HEADER_LENGTH){
    throw InvalidHeaderBuffer
  }

  header.Version = parseInt(buffer.toString(ENCODING, 0, 8).trimRight())
  header.Patient = buffer.toString(ENCODING, 8, 88).trimRight()
  header.Id = buffer.toString(ENCODING, 88, 168).trimRight()
    // todo: convert to date time
  const sdate = buffer.toString(ENCODING, 168, 176).trimRight()
  const stime = buffer.toString(ENCODING, 176, 184).trimRight()

  header.Start = parse_edfstart(
        sdate, stime
    )

  header.HeaderLength = parseInt(buffer.toString(ENCODING, 184, 192).trimRight())
  header.Reserved = buffer.toString(ENCODING, 192, 236).trimRight()
  header.DataRecords = parseInt(buffer.toString(ENCODING, 236, 244).trimRight())
    // convert to duration
  header.Duration = buffer.toString(ENCODING, 244, 252).trimRight()
  header.Signals = parseInt(buffer.toString(ENCODING, 252).trimRight())
  return true
}

function parse_edfstart (sdate, stime) {
  const date_parts = sdate.split(DATE_SEP)
  const time_parts = stime.split(DATE_SEP)

  if (date_parts.length !== 3 || time_parts.length !== 3) {
    throw InvalidStart
  }

  var year = parseInt(date_parts[2])
  const clipping = (year >= 85 && year <= 99) ? 1900 : 2000

  return new Date(
        year + clipping, date_parts[1] - 1, date_parts[0],
        time_parts[0], time_parts[1], time_parts[2]
    )
}

function read_signal_headers (status, data) {
    // initialize
  if (!status.signals) {
        // todo: may be find a more functional approarch to
        // initialize the array
    status.signals = []
    for (let i = 0; i < status.header.Signals; i++) {
      status.signals.push({})
    }

        // signal marker
    status.buffer[HEADER_LENGTH - 1] = 0
        // field marker
    status.buffer[HEADER_LENGTH - 2] = 0
  }

  var current_field = status.buffer[HEADER_LENGTH - 2]
  var current_signal = status.buffer[HEADER_LENGTH - 1]
  var waterMark = status.waterMark
  var used = 0

  for (let i = current_field; i < signal_header_fields.length; i++) {
    const field = signal_header_fields[i]
    const signals = status.signals.slice(current_signal)

    const result = parse_signal_header_field(
            field, signals, data.slice(used), status.buffer, waterMark
        )

    used += result.read

    if (
        result.waterMark !== 0 ||
        status.header.Signals !== (current_signal + result.parsed)
    ) {
      status.waterMark = result.waterMark
      status.buffer[HEADER_LENGTH - 1] = current_signal + result.parsed
      status.buffer[HEADER_LENGTH - 2] = i
      return false
    }

    current_signal = 0
    waterMark = result.waterMark
  }

  return used
}

function parse_signal_header_field (field, signals, data, scratch_pad, waterMark) {
  const {name, length: FIELD_SIZE, cast = null} = field
  const SIGNAL_TOTAL = signals.length
  var buffer = scratch_pad.slice(0, FIELD_SIZE)
  var i = 0
  var read = 0

  while (i < SIGNAL_TOTAL) {
    let to_copy = FIELD_SIZE - waterMark

    let copied = data.copy(
            buffer, waterMark, read, Math.min(read + to_copy, data.length)
        )

    waterMark += copied
    read += copied

    if (FIELD_SIZE !== waterMark) {
      break
    }

    let value = buffer.toString(ENCODING).trimRight()
    if (cast) {
      value = cast(value)
    }

    signals[i][name] = value
    i++
    waterMark = 0
  }

  return {
    read: read,
    waterMark: waterMark,
    parsed: i
  }
}

function read_data_block (status, data) {
    // initialize
  if (!status.block) {
    const block_size = status.signals.reduce((acc, signal) => {
      return acc + signal.Samples * 2
    }, 0)
    log(block_size)
    status.block = Buffer.allocUnsafe(block_size)
  }

  var waterMark = status.waterMark
  const to_copy = status.block.length - waterMark
  const copied = data.copy(
        status.block, waterMark, 0, Math.min(to_copy, data.length)
    )

  waterMark += copied
  status.waterMark = (waterMark === status.block.length) ? 0 : waterMark
  if (!status.waterMark) {
    log('Block Parsed: ' + status.block.length)
  }

  return {
    used: copied,
    parsed: (status.waterMark === 0)
  }
}

exports = module.exports = {
  DataBlocks,
  ParseEdfHeader
}
