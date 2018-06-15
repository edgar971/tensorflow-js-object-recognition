import * as tf from '@tensorflow/tfjs'
import Webcam from './media/webcam'
import { getTopKClasses } from './utils'

const REMOTE_MODEL_URL = 'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json'
const LOCAL_MODEL_URL = 'indexeddb://mobilenet-model'

export default async () => {
  const startBtn = document.getElementById('start')
  const stopBtn = document.getElementById('stop')
  const labelsElmt = document.getElementById('labels')
  const webcam = new Webcam(document.getElementById('webcam'))
  
  let shouldPredict = false
  let mobilenet

  try {
    mobilenet = await tf.loadModel(LOCAL_MODEL_URL);
  } catch {
    await tf.io.copyModel(REMOTE_MODEL_URL, LOCAL_MODEL_URL)
    mobilenet = await tf.loadModel(LOCAL_MODEL_URL);
  }

  startBtn.addEventListener('click', async (e) => {
    shouldPredict = true
    while (shouldPredict) {
      const labels = await predict(mobilenet, webcam)
      labelsElmt.innerHTML = `${labels.map(l => l.className)}`
    }
  })

  stopBtn.addEventListener('click', (e) => {
    shouldPredict = false
  })

  await webcam.setup()
}

async function predict(model, webcam) {
  const predictions = tf.tidy(() => {
    return model.predict(webcam.capture())
  })

  const labels = await getTopKClasses(predictions, 10)
  predictions.dispose()
  await tf.nextFrame()

  return labels
}
