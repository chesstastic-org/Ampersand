use std::fs;

use serde;
use serde::Deserialize;
use serde_json;

use crate::engine::search_info::SearchInfo;


#[derive(Deserialize)]
pub struct BaseNNUEInfo {
    #[serde(rename = "linear.0.weight")]
    pub hidden_weights: Vec<Vec<f64>>,
    #[serde(rename = "linear.0.bias")]
    pub hidden_biases: Vec<f64>,
    #[serde(rename = "linear.2.weight")]
    pub output_weights: Vec<Vec<f64>>,
    #[serde(rename = "linear.2.bias")]
    pub output_biases: Vec<f64>
}

pub fn quantize(val: &f64) -> i16 {
    (val * 64.) as i16
}

pub struct NNUE {
    pub input_size: usize,
    pub hidden: Layer,
    pub output: Layer
}

pub fn parse_weights(weights: Vec<Vec<f64>>) -> Vec<Vec<i16>> {
    weights.iter()
        .map(|weights| weights.iter().map(quantize).collect::<Vec<_>>())
        .collect::<Vec<_>>()
}

pub fn parse_biases(biases: Vec<f64>) -> Vec<i16> {
    biases.iter().map(quantize).collect::<Vec<_>>()
}

pub fn load_nnue(path: &str) -> NNUE {
    let text = fs::read_to_string(path).expect("Could not read model weights");
    let info: BaseNNUEInfo = serde_json::from_str(&text).expect("Could not parse model JSON");

    let hidden_weights = parse_weights(info.hidden_weights);
    let hidden_biases = parse_biases(info.hidden_biases);
    let output_weights = parse_weights(info.output_weights);
    let output_biases = parse_biases(info.output_biases);

    let hidden_neurons = hidden_weights.iter().zip(hidden_biases).map(|(weights, bias)| Neuron {
        weights: weights.clone(),
        bias
    }).collect::<Vec<_>>();
    let output_neurons = output_weights.iter().zip(output_biases).map(|(weights, bias)| Neuron {
        weights: weights.clone(),
        bias
    }).collect::<Vec<_>>();

    NNUE {
        input_size: hidden_neurons[0].weights.len(),
        hidden: Layer(hidden_neurons),
        output: Layer(output_neurons)
    }
}

pub fn relu(val: i16) -> i16 {
    if val >= 0 {
        val
    } else {
        0
    }
}

pub fn identity(val: i16) -> i16 {
    val
}

pub struct Neuron {
    pub weights: Vec<i16>,
    pub bias: i16
}

pub struct Layer(Vec<Neuron>);

fn process_neuron(input: &[i16], neuron: &Neuron) -> i16 {
    let mut sum = 0;
    for (val, weight) in input.iter().zip(&neuron.weights) {
        sum += weight * val;
    }
    sum + neuron.bias
}

pub fn update_neuron(sum: &mut i16, added_features: &[u16], removed_features: &[u16], neuron: &Neuron) {
    for ind in added_features {
        *sum += neuron.weights[*ind as usize];
    }
    for ind in removed_features {
        *sum -= neuron.weights[*ind as usize];
    }
}

fn apply_layer(input: &[i16], output: &mut [i16], layer: &Layer, activation: impl Fn(i16) -> i16) {
    layer.0.iter()
        .map(|neuron| process_neuron(&input, &neuron))
        .map(activation)
        .enumerate()
        .for_each(|(ind, neuron) | { output[ind] = neuron; });
}

pub fn update_layer(input: &mut [i16], added_features: &[u16], removed_features: &[u16], layer: &Layer, activation: impl Fn(i16) -> i16) {
    for (ind, neuron) in layer.0.iter().enumerate() {
        update_neuron(&mut input[ind], added_features, removed_features, neuron);
        input[ind] = activation(input[ind]);
    }
}

pub fn apply_hidden<'a, const T: usize>(info: &mut SearchInfo<T>) {
    let (input, rest) = info.layers.split_at_mut(1);
    let (hidden, _) = rest.split_at_mut(1);
    let input = &mut input[0];
    let hidden = &mut hidden[0];
    apply_layer(input, hidden, &info.nnue.hidden, relu);
}

pub fn update_hidden<'a, const T: usize>(info: &mut SearchInfo<T>, added_features: &[u16], removed_features: &[u16], activation: impl Fn(i16) -> i16) {
    let hidden = &mut info.layers[1];
    update_layer(hidden, added_features, removed_features, &info.nnue.hidden, activation);
}

pub fn eval_nnue<'a, const T: usize>(info: &mut SearchInfo<T>) -> Vec<i16> {
    let (_, rest) = info.layers.split_at_mut(1);
    let (hidden, rest) = rest.split_at_mut(1);
    let hidden = &mut hidden[0];
    let output = &mut rest[0];
    apply_layer(hidden, output, &info.nnue.output, identity);
    output.clone()
}

pub fn alloc_layers(nnue: &NNUE) -> Vec<Vec<i16>> {
    vec![
        vec![0; nnue.input_size],
        vec![0; nnue.hidden.0.len()],
        vec![0; nnue.output.0.len()]
    ]
}