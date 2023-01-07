pub mod node;

pub mod layers_structure {
    use libm::sqrt;
    use ndarray::{Array, Array1, Array2, ArrayView1};
    use rand::distributions::Distribution;
    use rand::thread_rng;
    use rand_distr::Normal;
    use std::iter::from_fn;

    #[derive(Debug)]
    pub enum LayerType {
        Convolutional,
        MaxPool,
        Sigmoid,
        SoftMax,
    }

    trait Default {
        fn default(prev_size: usize, size: usize) -> Self;
    }

    pub trait RandomGeneration {
        fn generate_number(mean: f32, deviation: f32) -> f32 {
            let n = Normal::new(mean, deviation).unwrap();
            n.sample(&mut thread_rng())
        }
        fn generate_range(size: usize, mean: f32, deviation: f32) -> Option<Array1<f32>> {
            match size {
                0 => None,
                _ => Some(
                    from_fn(|| Some(Self::generate_number(mean, deviation)))
                        .take(size)
                        .collect(),
                ),
            }
        }
    }

    #[derive(Debug)]
    pub struct Layer {
        pub size: usize,
        pub weights: Array2<f32>,
        pub biases: Array1<f32>,
        pub z_vec: Array1<f32>,
        pub layer_type: LayerType,
    }

    impl RandomGeneration for LayerType {}

    impl LayerType {
        pub fn generate_weights(&self, prev_size: usize, cur_size: usize) -> Array2<f32> {
            match self {
                LayerType::Convolutional => {
                    let a = Array2::<f32>::zeros((5, 5));

                    Array2::from_shape_vec(
                        (5, 5),
                        a.into_iter()
                            .map(|_| {
                                LayerType::generate_number(0.0, 1.0 / sqrt(prev_size as f64) as f32)
                            })
                            .collect(),
                    )
                    .unwrap()
                }
                LayerType::MaxPool => Array2::default((0, 0)),
                LayerType::Sigmoid => {
                    let a = Array2::<f32>::zeros((prev_size, cur_size));
                    Array2::from_shape_vec(
                        a.raw_dim(),
                        a.into_iter()
                            .map(|_| {
                                LayerType::generate_number(0.0, 1.0 / sqrt(prev_size as f64) as f32)
                            })
                            .collect(),
                    )
                    .unwrap()
                }
                LayerType::SoftMax => Array2::<f32>::zeros((cur_size, prev_size)),
            }
        }

        pub fn generate_bias(&self, prev_size: usize, cur_size: usize) -> Array1<f32> {
            match self {
                LayerType::Convolutional => {
                    LayerType::generate_range(1, 0.0, 1.0 / sqrt(prev_size as f64) as f32).unwrap()
                }
                LayerType::MaxPool => Array1::default(0),
                LayerType::Sigmoid => LayerType::generate_range(cur_size, 0.0, 1.0).unwrap(),
                LayerType::SoftMax => Array::zeros(cur_size),
            }
        }
    }

    impl Layer {
        pub fn delta(&self, out: &ArrayView1<f32>, desired_out: &Array1<f32>) -> Array2<f32> {
            let v: Array1<f32> = out.to_owned() - desired_out.to_owned();
            Array2::from_shape_vec((v.len(), 1), v.into_raw_vec()).unwrap()
        }
        pub fn new(prev_size: usize, cur_size: usize, layer_type: LayerType) -> Layer {
            Layer {
                size: cur_size,
                biases: layer_type.generate_bias(prev_size, cur_size).t().to_owned(),
                weights: layer_type
                    .generate_weights(prev_size, cur_size)
                    .t()
                    .to_owned(),
                z_vec: Array1::zeros(cur_size),
                layer_type,
            }
        }

        //#[derive(Debug, Clone)]
        /*pub struct LayerBias {
            size: usize,
            pub nodes: Vec<Node>,
            d_e: Vec<f32>,
        }

        #[derive(Debug, Clone)]
        pub struct LayerDigit {
            prev_size: usize,
            size: usize,
            pub nodes: Vec<Node>,
            cost: Vec<f32>,
            d_e: Vec<f32>,
        }

        impl Default for LayerBias {
            fn default(prev_size: usize, size: usize) -> LayerBias {
                LayerBias {
                    size,
                    nodes: from_fn(|| Some(Node::new(prev_size))).take(size).collect(),
                    d_e: {
                        let mut d_e = Vec::new();
                        d_e.resize(prev_size, 0.0);
                        d_e
                    },
                }
            }
        }

        impl Default for LayerDigit {
            fn default(prev_size: usize, size: usize) -> Self {
                LayerDigit {
                    prev_size,
                    size: 10,
                    nodes: from_fn(|| Some(Node::new(prev_size))).take(size).collect(),
                    cost: Vec::new(),
                    d_e: {
                        let mut d_e = Vec::<f32>::new();
                        d_e.resize(10, 0.0);
                        d_e
                    },
                }
            }
        }

        pub trait BackPropagation {
            fn get_weights(&self, index: usize) -> Box<Vec<f32>>;
            fn get_errors(&self, index: usize) -> Box<Vec<f32>>;
            fn get_bias(&mut self, index: usize) -> &mut f32;
            fn get_nodes(&self) -> &Vec<Node>;
            fn get_dE(&self, index: usize) -> &Vec<f32>;
            fn get_input(&self, index: usize) -> &Vec<f32>;
            fn change_weight(&mut self);
            fn change_bias(&mut self);
        }

        /*pub trait Layer {
            fn new(prev_size: usize, size: usize) -> Self;
            fn fill_input(&mut self, input: Arc<Vec<f32>>);
        }*/

        impl Layer for LayerBias {
            fn new(prev_size: usize, size: usize) -> LayerBias {
                Default::default(prev_size, size)
            }

            fn fill_input(&mut self, input: Arc<Vec<f32>>) {
                self.nodes.par_iter_mut().for_each(|x| {
                    x.input = input.clone();
                });
            }
        }

        impl Layer for LayerDigit {
            fn new(prev_size: usize, size: usize) -> Self {
                Default::default(prev_size, size)
            }

            fn fill_input(&mut self, input: Arc<Vec<f32>>) {
                self.nodes.par_iter_mut().for_each(|x| {
                    x.input = input.clone();
                });
            }
        }

        impl BackPropagation for LayerBias {
            fn get_weights(&self, index: usize) -> Box<Vec<f32>> {
                self.nodes.get(index).unwrap().weight.clone()
            }

            fn get_errors(&self, index: usize) -> Box<Vec<f32>> {
                self.nodes.get(index).unwrap().error.clone()
            }

            fn get_bias(&mut self, index: usize) -> &mut f32 {
                &mut self.nodes.get_mut(index).unwrap().bias
            }

            fn get_input(&self, index: usize) -> &Vec<f32> {
                &self.nodes.get(index).unwrap().input
            }

            fn get_nodes(&self) -> &Vec<Node> {
                &self.nodes
            }

            fn get_dE(&self, index: usize) -> &Vec<f32> {
                &self.d_e
            }

            fn change_weight(&mut self) {
                self.nodes.par_iter_mut().for_each(|x| {
                    x.weight.iter_mut().for_each(|y| {
                        *y -= x
                            .input
                            .iter()
                            .enumerate()
                            .map(|z| *z.1 * x.error.get(z.0).unwrap())
                            .sum::<f32>()
                    })
                });
            }
            fn change_bias(&mut self) {
                todo!()
            }
        }

        impl BackPropagation for LayerDigit {
            fn get_weights(&self, index: usize) -> Box<Vec<f32>> {
                self.nodes.get(index).unwrap().weight.clone()
            }

            fn get_errors(&self, index: usize) -> Box<Vec<f32>> {
                self.nodes.get(index).unwrap().error.clone()
            }

            fn get_bias(&mut self, index: usize) -> &mut f32 {
                &mut self.nodes.get_mut(index).unwrap().bias
            }

            fn get_nodes(&self) -> &Vec<Node> {
                &self.nodes
            }

            fn change_weight(&mut self) {
                self.nodes.iter_mut().for_each(|x| {
                    x.weight
                        .iter_mut()
                        .enumerate()
                        .for_each(|y| *y.1 -= x.error.get(y.0).unwrap())
                });
            }
            fn change_bias(&mut self) {
                todo!()
            }

            fn get_input(&self, index: usize) -> &Vec<f32> {
                &self.nodes.get(index).unwrap().input
            }

            fn get_dE(&self, index: usize) -> &Vec<f32> {
                &self.d_e
            }
        }

        impl LayerBias {
            pub fn sigmoid(&mut self) {
                self.nodes.par_iter_mut().for_each(|x| {
                    x.z = -(x
                        .weight
                        .iter()
                        .enumerate()
                        .map(|y| y.1 * x.input.get(y.0).unwrap())
                        .sum::<f32>()
                        - x.bias);
                    x.output = 1.0 / (1.0 + exp(x.z as f64)) as f32;
                });
            }

            pub fn error_generate(
                &mut self,
                back_layer: &dyn BackPropagation,
                next_layer: &dyn BackPropagation,
            ) {
                self.nodes.iter_mut().enumerate().for_each(|x| {
                    x.1.error.iter_mut().enumerate().for_each(|z| {
                        *z.1 = next_layer
                            .get_dE(z.0)
                            .iter()
                            .enumerate()
                            .map(|y| {
                                *self.d_e.get_mut(z.0).unwrap() =
                                    *next_layer.get_weights(y.0).get(y.0).unwrap() * *y.1;
                                *self.d_e.get(z.0).unwrap()
                            })
                            .sum::<f32>()
                            * back_layer.get_input(z.0).get(z.0).unwrap()
                            * next_layer.get_dE(x.0).get(x.0).unwrap();
                    });
                })
                /*self.nodes.iter_mut().enumerate().for_each(|x| {
                    *x.1.error.get_mut(x.0).unwrap() =
                });*/
                /*next_layer.iter_mut().for_each(|x| x.error)
                self.back_error.resize(errors.len(), 0.0);
                let mut it = next_layer.iter();
                let mut error = errors.iter();*/
                /*self.back_error.iter_mut().enumerate().for_each(|y| {
                    *y.1 = *it.next().unwrap().weight.get(y.0).unwrap() * error.next().unwrap()
                });
                self.back_error.iter_mut().enumerate().for_each(|y| {
                    let pos = it.next().unwrap();
                    *y.1 = errors.get(y.0).unwrap() / (pos.output * pos.weight.get(y.0).unwrap())
                        + pos.weight.get(y.0).unwrap();
                });*/
            }
        }

        impl LayerInput {
            pub(crate) fn new(input: &Vec<f32>) -> LayerInput {
                LayerInput {
                    size: input.len(),
                    input: Arc::new(input.clone()),
                }
            }
        }

        impl LayerDigit {
            fn cost_function(&mut self, input: Vec<f32>) {
                self.cost = self
                    .nodes
                    .iter_mut()
                    .enumerate()
                    .map(|x| x.1.output - input[x.0])
                    .collect();
            }

            fn error_generate(&mut self) {
                self.cost_function();
                self.nodes.iter_mut().enumerate().for_each(|x| {
                    x.1.error.iter_mut().enumerate().for_each(|y| {
                        *y.1 = self.cost.get(x.0).unwrap()
                            * x.1.output
                            * (1.0 - x.1.output)
                            * x.1.input.get(x.0).unwrap();
                        *self.d_e.get_mut(x.0).unwrap() =
                            self.cost.get(x.0).unwrap() * x.1.output * (1.0 - x.1.output);
                    })
                });
            }

            pub fn sigmoid(&mut self) {
                self.nodes.par_iter_mut().for_each(|x| {
                    x.z = -((x
                        .weight
                        .iter()
                        .enumerate()
                        .map(|y| y.1 * x.input.get(y.0).unwrap()))
                    .sum::<f32>()
                        - x.bias);
                    x.output = 1.0 / (1.0 + exp(x.z as f64)) as f32;
                });
                self.error_generate();
            }
        }*/
    }
}
