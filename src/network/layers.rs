pub mod node;

pub mod layers_structure {
    use crate::network::layers::node::nodes::NodeIO;
    use std::iter::from_fn;

    trait Default {
        fn default(size: usize) -> Self;
    }

    #[derive(Debug)]
    pub struct LayerBias {
        size: usize,
        input: Vec<f64>,
        nodes: Vec<NodeIO>,
    }

    impl Default for LayerBias {
        fn default(size: usize) -> LayerBias {
            LayerBias {
                size,
                input: Vec::new(),
                nodes: from_fn(|| Some(NodeIO::new(size))).take(size).collect(),
            }
        }
    }

    pub trait Layer {
        fn new(size: usize) -> Self;
        fn generate_weight(&mut self);
        fn fill_input(&mut self, input: Vec<f64>);
    }

    impl Layer for LayerBias {
        fn new(size: usize) -> LayerBias {
            Default::default(size)
        }
        fn generate_weight(&mut self) {}
        fn fill_input(&mut self, input: Vec<f64>) {
            self.input = input;
        }
    }
}
