const assert = require('assert')
const MemoryStream = require('memorystream')
const {ParseEdfHeader} = require('../src/main.js').utils
const {DataBlocks} = require('../src/main.js')

describe('ParseEdfHeader', function () {
  it('Throw exception on invalid buffer', function () {
    const raw = '0       X X X X'
    var header = {}

    assert.throws(
            () => {
              const result = ParseEdfHeader(
                Buffer.from(raw, 'ascii'),
                header
              )
            },
            Error, 'Edf header encoded can no be less than 256'
        )
  })

  it('Parsing', function () {
    const expected_header = {
      Version: 0,
      Patient: 'X X X X',
      Id: 'Startdate 12-AUG-2009 X X BCI2000',
      Start: new Date('2009-08-12T20:15:00.000Z'),
      HeaderLength: 16896,
      Reserved: 'EDF+C',
      DataRecords: 61,
      Duration: '1',
      Signals: 65
    }

    const raw = '0       X X X X                                                                         Startdate 12-AUG-2009 X X BCI2000                                               12.08.0916.15.0016896   EDF+C                                       61      1       65  '

    var header = {}
    const result = ParseEdfHeader(
            Buffer.from(raw, 'ascii'),
            header
        )

    assert.equal(true, result, 'Parsed')
    assert.deepStrictEqual(expected_header, header, 'Successfully parsed')
  })
})

describe('DataBlock Parsing header', function () {
  it('Stream splitted header', function (done) {
    var input = MemoryStream.createReadStream([
      '0       X X ',
      'X X                     ',
      '                ',
      '                 ',
      '                   Start',
      'date 12-AUG-2009 X ',
      'X BCI2000             ',
      '                ',
      '                  12.08.0916.',
      '15.0016896   EDF+',
      'C                         ',
      '              61      1       65  '
    ]
    )

    const expected_header = {
      Version: 0,
      Patient: 'X X X X',
      Id: 'Startdate 12-AUG-2009 X X BCI2000',
      Start: new Date('2009-08-12T20:15:00.000Z'),
      HeaderLength: 16896,
      Reserved: 'EDF+C',
      DataRecords: 61,
      Duration: '1',
      Signals: 65
    }

    const blocks = new DataBlocks()
    var cb_called = false
    blocks.on('header', (header) => {
      cb_called = true
      assert.deepStrictEqual(header, expected_header, 'Successfully parsed')
    })

    input.on('end', () => {
      // confirm header callback was call
      assert.equal(true, cb_called, 'header event has been trigger')
      done()
    })

    input.pipe(blocks)
    // assert.equal(true,false)
  })

  it('Specifying Header parser', function (done) {
    var input = MemoryStream.createReadStream([
      '0       X X ',
      'X X                     ',
      '                ',
      '                 ',
      '                   Start',
      'date 12-AUG-2009 X ',
      'X BCI2000             ',
      '                ',
      '                  12.08.0916.',
      '15.0016896   EDF+',
      'C                         ',
      '              61      1       65  '
    ]
    )

    const expected_header = {
      Version: 0,
      Patient: 'X X X X',
      Id: 'Startdate 12-AUG-2009 X X BCI2000',
      Start: new Date('2009-08-12T20:15:00.000Z'),
      HeaderLength: 16896,
      Reserved: 'EDF+C',
      DataRecords: 61,
      Duration: '1',
      Signals: 65
    }

    var parser_called = false
    var headerParser = function (buffer, header) {
      parser_called = true
      return ParseEdfHeader(buffer, header)
    }

    const blocks = new DataBlocks({
      headerParser: headerParser
    })

    var cb_called = false
    blocks.on('header', (header) => {
      cb_called = true
      assert.deepStrictEqual(header, expected_header, 'Successfully parsed')
    })

    input.on('end', () => {
      // confirm header callback was call
      assert.equal(true, parser_called, 'header parser have been call')
      assert.equal(true, cb_called, 'header event has been trigger')
      done()
    })

    input.pipe(blocks)
    // assert.equal(true,false)
  })

  it('Stream Signal headers', function (done) {
    var input = MemoryStream.createReadStream([
      '0       MCH-0234567 F 16-SEP-1987 Haagse_Harry                                          Startdate 16-SEP-1987 PSG-1234/1987 NN Telemetry03                              16.09.8720.35.00768     Reserved field of 44 characters             2880    30      2   ',
      'EEG Fpz-Cz  ',
      '    Temp rectal     AgAgCl cup electro',
      'des                 ',
      '                                          Rectal ',
      'thermist',
      'o',
      'r', '                                                               uV      ',
      'degC    -440    34.4    510     40.2    -',
      '2048   -2048   2047    2047    HP:0.1Hz LP:',
      '75Hz N:50Hz                                                         LP:0.1Hz (first order)                                                          15000   3       Reserved for EEG signal         Reserved for Body temperature   '
    ])

    const expected_header = {
      Version: 0,
      Patient: 'MCH-0234567 F 16-SEP-1987 Haagse_Harry',
      Id: 'Startdate 16-SEP-1987 PSG-1234/1987 NN Telemetry03',
      Start: new Date('1987-09-17T00:35:00.000Z'),
      HeaderLength: 768,
      Reserved: 'Reserved field of 44 characters',
      DataRecords: 2880,
      Duration: '30',
      Signals: 2
    }

    const expected_signals = [{
      Label: 'EEG Fpz-Cz',
      Transducer: 'AgAgCl cup electrodes',
      PhysicalDimensions: 'uV',
      PhysicalMin: -440,
      PhysicalMax: 510,
      DigitalMin: -2048,
      DigitalMax: 2047,
      Prefiltering: 'HP:0.1Hz LP:75Hz N:50Hz',
      Samples: 15000,
      Reserved: 'Reserved for EEG signal'
    }, {
      Label: 'Temp rectal',
      Transducer: 'Rectal thermistor',
      PhysicalDimensions: 'degC',
      PhysicalMin: 34.4,
      PhysicalMax: 40.2,
      DigitalMin: -2048,
      DigitalMax: 2047,
      Prefiltering: 'LP:0.1Hz (first order)',
      Samples: 3,
      Reserved: 'Reserved for Body temperature'
    }
    ]

    const blocks = new DataBlocks()
    var cb_count = 0

    blocks.on('header', (header) => {
      cb_count += 1
      assert.deepStrictEqual(header, expected_header, 'Successfully header parsed')
    })

    blocks.on('signals', (signals) => {
      cb_count += 1
      assert.deepStrictEqual(signals, expected_signals, 'Successfully signal parsed')
    })

    input.on('end', () => {
      // confirm header callback was call
      assert.equal(2, cb_count, 'Both header and signal events has been triggered')
      done()
    })

    input.pipe(blocks)
  })
})
