import * as tf from '@tensorflow/tfjs'


// Creating models
(async () => {

  const model = tf.sequential()

  const hidden = tf.layers.dense({
    inputShape: [2],
    units: 4,
    activation: 'sigmoid'
  })

  model.add(hidden)

  const output = tf.layers.dense({
    units: 1,
    activation: 'sigmoid'
  })

  model.add(output)

  model.compile({
    optimizer: tf.train.sgd(0.1),
    loss: 'meanSquaredError'
  })

  const xs = tf.tensor2d([
    [0,0],
    [.5, .5],
    [1, 1]
  ])

  const ys = tf.tensor2d([
    [1],
    [0.5],
    [0]
  ]);

  for (let index = 0; index < 300; index++) {
    const { history } = await model.fit(xs, ys, { shuffle: true, epochs: 10 })
    console.log(history.loss)
  }

  console.log('trainig done')

  let prediction = model.predict(xs)

  prediction.print()
})()