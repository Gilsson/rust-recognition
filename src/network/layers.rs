pub mod node;

pub mod layers_structure {
    use crate::network::layers::node::nodes::NodeIO;
    use libm::{exp, pow, powf};
    use std::borrow::Borrow;
    use std::iter::from_fn;
    use std::rc::Rc;

    trait Default {
        fn default(prev_size: usize, size: usize) -> Self;
    }

    #[derive(Debug)]
    pub struct LayerBias {
        size: usize,
        pub nodes: Vec<NodeIO>,
        back_error: Vec<f64>,
    }

    #[derive(Debug)]
    pub struct LayerInput {
        size: usize,
        pub input: Vec<f64>,
        back_error: Vec<f64>,
    }

    #[derive(Debug)]
    pub struct LayerDigit {
        prev_size: usize,
        size: usize,
        pub nodes: Vec<NodeIO>,
        expected_output: Vec<f64>,
        cost: Vec<f64>,
        back_error: Vec<f64>,
    }

    impl Default for LayerBias {
        fn default(prev_size: usize, size: usize) -> LayerBias {
            LayerBias {
                size,
                nodes: from_fn(|| Some(NodeIO::new(prev_size)))
                    .take(size)
                    .collect(),
                back_error: Vec::new(),
            }
        }
    }

    impl Default for LayerDigit {
        fn default(prev_size: usize, size: usize) -> Self {
            LayerDigit {
                prev_size,
                size: 10,
                nodes: from_fn(|| Some(NodeIO::new(prev_size)))
                    .take(size)
                    .collect(),
                expected_output: vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                cost: Vec::new(),
                back_error: Vec::new(),
            }
        }
    }

    pub trait BackPropagation {
        fn get_weights(&mut self, index: usize) -> &mut Vec<f64>;
        fn get_errors(&self) -> &Vec<f64>;
        fn get_bias(&mut self, index: usize) -> &mut f64;
        fn get_nodes(&self) -> &Vec<NodeIO>;
        fn change_weight(&mut self);
        fn change_bias(&mut self);
    }

    pub trait Layer {
        fn new(prev_size: usize, size: usize) -> Self;
        fn fill_input(&mut self, input: &Vec<f64>);
    }

    impl Layer for LayerBias {
        fn new(prev_size: usize, size: usize) -> LayerBias {
            Default::default(prev_size, size)
        }

        fn fill_input(&mut self, input: &Vec<f64>) {
            self.nodes.iter_mut().for_each(|x| {
                input
                    .iter()
                    .enumerate()
                    .map(|y| x.input[y.0] = *y.1)
                    .collect()
            });
        }
    }

    impl Layer for LayerDigit {
        fn new(prev_size: usize, size: usize) -> Self {
            Default::default(prev_size, size)
        }

        fn fill_input(&mut self, input: &Vec<f64>) {
            self.nodes.iter_mut().for_each(|x| {
                input
                    .iter()
                    .enumerate()
                    .map(|y| x.input[y.0] = *y.1)
                    .collect()
            });
        }
    }

    impl BackPropagation for LayerBias {
        fn get_weights(&mut self, index: usize) -> &mut Vec<f64> {
            &mut self.nodes.get_mut(index).unwrap().weight
        }

        fn get_errors(&self) -> &Vec<f64> {
            &self.back_error
        }

        fn get_bias(&mut self, index: usize) -> &mut f64 {
            &mut self.nodes.get_mut(index).unwrap().bias
        }

        fn get_nodes(&self) -> &Vec<NodeIO> {
            &self.nodes
        }
        fn change_weight(&mut self) {
            todo!()
        }
        fn change_bias(&mut self) {
            todo!()
        }
    }

    impl BackPropagation for LayerDigit {
        fn get_weights(&mut self, index: usize) -> &mut Vec<f64> {
            &mut self.nodes.get_mut(index).unwrap().weight
        }

        fn get_errors(&self) -> &Vec<f64> {
            &self.back_error
        }

        fn get_bias(&mut self, index: usize) -> &mut f64 {
            &mut self.nodes.get_mut(index).unwrap().bias
        }

        fn get_nodes(&self) -> &Vec<NodeIO> {
            &self.nodes
        }

        fn change_weight(&mut self) {
            todo!()
        }
        fn change_bias(&mut self) {
            todo!()
        }
    }

    impl LayerBias {
        pub fn sigmoid(&mut self) {
            self.nodes.iter_mut().for_each(|x| {
                x.z = -(x.weight.iter().sum::<f64>() * x.input.iter().sum::<f64>()) - x.bias;
                x.output = 1.0 / (1.0 + exp(x.z));
            });
        }

        pub fn error_generate(&mut self, next_layer: &[NodeIO], errors: &[f64]) {
            //let errors = next_layer.get_errors().clone();
            self.back_error.resize(10, 0.0);
            let mut it = next_layer.iter();
            self.back_error.iter_mut().enumerate().for_each(|y| {
                let pos = it.next().unwrap();
                *y.1 = pow(pos.output, 2.0)
                    * exp(pos.z)
                    * errors.get(y.0).unwrap()
                    * pos.weight.get(y.0).unwrap();
                /*next_layer
                //.get_nodes()
                .iter()
                .enumerate()
                .map(|x| {
                    *y = (errors.get(x.0).unwrap()
                        * exp(x.1.z)
                        * x.1.output
                        * x.1.output
                        * x.1.weight.get(x.0).unwrap());
                    *y
                }).collect::<Vec<f64>>();*/
            });
        }
    }

    impl LayerInput {
        pub(crate) fn new(input: Vec<f64>) -> LayerInput {
            LayerInput {
                size: input.len(),
                input,
                back_error: Vec::new(),
            }
        }
    }

    impl LayerDigit {
        fn cost_function(&mut self) {
            self.cost = self
                .nodes
                .iter_mut()
                .enumerate()
                .map(|x| x.1.output - self.expected_output[x.0])
                .collect();
        }

        fn error_generate(&mut self) {
            self.cost_function();
            self.back_error = self
                .nodes
                .iter_mut()
                .enumerate()
                .map(|x| self.cost.get(x.0).unwrap() * exp(x.1.z) * x.1.output * x.1.output)
                .collect();
        }

        pub fn sigmoid(&mut self) {
            self.nodes.iter_mut().for_each(|x| {
                x.z = -(x.weight.iter().sum::<f64>() * x.input.iter().sum::<f64>()) - x.bias;
                x.output = 1.0 / (1.0 + exp(x.z));
            });
            self.error_generate();
        }
    }
}
