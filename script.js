// ? Note: 6 tensors were created. 2 tensors were created when I loaded the model itself. Then 2 tensors were made for the inputs and two tensors were created but were returned to me as outputs.

// ! Reminder: this code prints to the console.

const MODEL_PATH =
  "https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/SavedModels/sqftToPropertyPrice/model.json";
let model = undefined;

// async function loadModel() {
//   model = await tf.loadLayersModel(MODEL_PATH);
//   model.summary();
// }

async function loadModel() {
  model = await tf.loadLayersModel(MODEL_PATH);
  model.summary();

  // batch 1
  const input = tf.tensor2d([[870]]);

  // batch 3
  const inputBatch = tf.tensor2d([[[500], [1100], [970]]]);

  // predictions for each batch
  const result = model.predict(input);
  const resultBatch = model.predict(inputBatch);

  // print results console
  result.print(); // can also use .array() to get results back as array
  resultBatch.print(); // or use .array() to get results back as array

  input.dispose();
  inputBatch.dispose();
  result.dispose();
  resultBatch.dispose();
  model.dispose();
}

loadModel();
