const fs = require('fs')
const edf = require('../src/main.js')
const Transform = require('stream').Transform

const file = 'S001R01.edf'

var reader = fs.createReadStream(file)


//pipe a DataRecord (edf.DataBlock) stream to the read stream representing the edf file
//pipe a RecordStream (edf.RecordsStream) to the above datarecord
//to process/translate each data record
//the function returns the final stream to be able to continue chaining.
var records = edf(reader)

//error handling
reader.on('error', (error) => {
  console.log(`Error: ${error}.`)
})

//error handling
records.on('error', (error) => {
  console.log(`Records Error: ${error}.`)
})

reader.on('end', () => {
  console.log(`Ending this`)
})

const objectToString = new Transform({
  writableObjectMode: true,
  transform (chunk, encoding, callback) {
    this.push(JSON.stringify(chunk) + '\n')
    callback()
  }
})

records.pipe(objectToString)
       .pipe(process.stdout)
