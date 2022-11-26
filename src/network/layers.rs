pub mod node;

pub mod layers_structure {
    use crate::network::layers::node::nodes::NodeIO;
<<<<<<< HEAD
    use libm::exp;
    use std::iter::from_fn;

    trait Default {
        fn default(prev_size: usize, size: usize) -> Self;
=======
    use std::iter::from_fn;

    trait Default {
        fn default(size: usize) -> Self;
>>>>>>> 2b34d59 (first commit)
    }

    #[derive(Debug)]
    pub struct LayerBias {
        size: usize,
        input: Vec<f64>,
        nodes: Vec<NodeIO>,
    }

<<<<<<< HEAD
    #[derive(Debug)]
    pub struct LayerInput {
        size: usize,
        input: Vec<f64>,
    }

    #[derive(Debug)]
    pub struct LayerDigit {
        prev_size: usize,
        size: usize,
        input: Vec<f64>,
        nodes: Vec<NodeIO>,
        output: i32,
        cost: Vec<f64>,
    }

    impl Default for LayerBias {
        fn default(prev_size: usize, size: usize) -> LayerBias {
            LayerBias {
                size,
                input: Vec::new(),
                nodes: from_fn(|| Some(NodeIO::new(prev_size)))
                    .take(size)
                    .collect(),
            }
        }
    }

    impl Default for LayerDigit {
        fn default(prev_size: usize, size: usize) -> Self {
            LayerDigit {
                prev_size: 0,
                size: 10,
                input: Vec::new(),
                nodes: from_fn(|| Some(NodeIO::new(prev_size)))
                    .take(size)
                    .collect(),
                output: -1,
                cost: Vec::new(),
=======
    impl Default for LayerBias {
        fn default(size: usize) -> LayerBias {
            LayerBias {
                size,
                input: Vec::new(),
                nodes: from_fn(|| Some(NodeIO::new(size))).take(size).collect(),
>>>>>>> 2b34d59 (first commit)
            }
        }
    }

    pub trait Layer {
<<<<<<< HEAD
        fn new(prev_size: usize, size: usize) -> Self;
=======
        fn new(size: usize) -> Self;
        fn generate_weight(&mut self);
>>>>>>> 2b34d59 (first commit)
        fn fill_input(&mut self, input: Vec<f64>);
    }

    impl Layer for LayerBias {
<<<<<<< HEAD
        fn new(prev_size: usize, size: usize) -> LayerBias {
            Default::default(prev_size, size)
        }
=======
        fn new(size: usize) -> LayerBias {
            Default::default(size)
        }
        fn generate_weight(&mut self) {}
>>>>>>> 2b34d59 (first commit)
        fn fill_input(&mut self, input: Vec<f64>) {
            self.input = input;
        }
    }
<<<<<<< HEAD

    impl Layer for LayerDigit {
        fn new(prev_size: usize, size: usize) -> Self {
            Default::default(prev_size, size)
        }

        fn fill_input(&mut self, input: Vec<f64>) {
            self.input = input;
        }
    }

    impl LayerBias {
        pub fn sigmoid(&mut self) {
            for it in &mut self.nodes {
                it.output = 1.0
                    / (1.0
                        + exp(
                            -(it.weight.iter().sum::<f64>() * it.input.iter().sum::<f64>())
                                - it.bias,
                        ))
            }
        }
    }

    impl LayerInput {
        pub(crate) fn new(input: Vec<f64>) -> LayerInput {
            LayerInput {
                size: input.len(),
                input,
            }
        }
    }

    impl LayerDigit {
        fn cost_function(&mut self) {}
        fn error_generate() {}
    }
=======
>>>>>>> 2b34d59 (first commit)
}
