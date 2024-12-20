FROM node:16

WORKDIR app
COPY . .

RUN mkdir /input
RUN mkdir /output
RUN npm install
RUN npm install --prefix ./module/edf-parser

CMD node convert.js
