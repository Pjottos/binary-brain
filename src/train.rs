use crate::*;
use rayon::prelude::*;
use rand::distributions::{Distribution, Uniform, Bernoulli};
use rand_xoshiro::Xoshiro256StarStar;
use std::cmp::Ordering;
use std::iter;

pub struct Genetic {
    population: Vec<(BinaryBrain, f64)>,
    p_mutate: f64,
    p_crossover: f64,
    pop_select: Uniform<usize>,
    tournament_size: usize,
    rng: Xoshiro256StarStar,
}

impl Genetic {
    pub fn new(initial: BinaryBrain, pop_size: usize, tournament_size: usize, mutation_chance: f64, crossover_switch_chance: f64) -> Result<Genetic> {
        if pop_size == 0 {
            return Err(BinaryBrainError::ZeroPopSize);
        }
        if tournament_size == 0 {
            return Err(BinaryBrainError::ZeroTournamentSize);
        }
        if mutation_chance < 0.0 || mutation_chance > 1.0 {
            return Err(BinaryBrainError::InvalidProbability(mutation_chance));
        }
        if crossover_switch_chance < 0.0 || crossover_switch_chance > 1.0 {
            return Err(BinaryBrainError::InvalidProbability(mutation_chance));
        }

        let mut pop = Vec::with_capacity(pop_size);
        pop.push((initial, f64::MIN));
        for _ in 0..pop_size - 1 {
            pop.push((BinaryBrain::from_template(&pop[0].0), f64::MIN));
        }

        Ok(Genetic {
            population: pop,
            p_mutate: mutation_chance,
            p_crossover: crossover_switch_chance,
            pop_select: Uniform::new(0, pop_size),
            tournament_size: tournament_size,
            rng: Xoshiro256StarStar::from_entropy(),
        })
    }

    pub fn evaluate<F: Fn(&mut BinaryBrain) -> f64 + Send + Sync>(&mut self, fitness: F) -> f64 {
        self.population.iter_mut().for_each(|p| {    
            p.1 = fitness(&mut p.0);
        });
        self.population.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal)).unwrap().1
    }

    pub fn evaluate_parallel<F: Fn(&mut BinaryBrain) -> f64 + Send + Sync>(&mut self, fitness: F) -> f64 {
        self.population.par_iter_mut().for_each(|p| {    
            p.1 = fitness(&mut p.0);
        });
        self.population.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal)).unwrap().1
    }

    pub fn breed(&mut self) {
        let pop_count = self.population.len();
        self.population = iter::repeat_with(|| {
            let parents: Vec<usize> = iter::repeat_with(|| {
                let count = self.tournament_size;
                let candidates = iter::repeat_with(|| {
                    let idx = self.pop_select.sample(&mut self.rng);
                    (idx, self.population[idx].1)
                }).take(count);
            
                candidates.max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal)).unwrap().0
            }).take(2).collect();

            // needed because we cannot mutate self while borrowing the brains
            let mut rng = self.rng.clone();
            self.rng.jump();
            let parents: Vec<&BinaryBrain> = parents.iter().map(|p| &self.population[*p].0).collect();

            // crossover + mutation
            let mut weights = Vec::with_capacity(parents[0].weights().len()); 
            let mut activations = Vec::with_capacity(parents[0].activations().len());
            
            let mut current_parent = 0;

            // weights
            {
                let mutate_in_seg = Bernoulli::new(1.0 - (1.0 - self.p_mutate).powi(64)).unwrap();
                let mutate = Bernoulli::new(self.p_mutate).unwrap();
                let crossover_in_seg = Bernoulli::new(1.0 - (1.0 - self.p_crossover).powi(64)).unwrap();
                let crossover_idx = Uniform::new(0, 64);
                for i in 0..parents[0].weights().len() {
                    let mut seg = parents[current_parent].weights()[i];
                    // check if crossover should occur in this segment, faster than checking on each bit seperately
                    if crossover_in_seg.sample(&mut rng) {
                        let idx = crossover_idx.sample(&mut rng);
                        // clear the bits [idx, 64)
                        let mask = u64::MAX.overflowing_shr(64 - idx).0;
                        seg &= mask;
                        // get the corresponding bits from the other parent
                        current_parent = if current_parent == 0 {1} else {0};
                        let seg2 = parents[current_parent].weights()[i];
                        seg |= seg2 & !mask;
                    }

                    if mutate_in_seg.sample(&mut rng) {
                        // with probability p_mutate, flip each bit in the segment
                        for (j, b) in mutate.sample_iter(&mut rng).take(64).enumerate() {
                            seg ^= (b as u64) << j;
                        }
                    }

                    weights.push(seg);
                }
            }

            // activations
            {
                // we treat each activation value as one parameter, so no need for
                // the segment manipulation done when copying the weights
                let crossover = Bernoulli::new(self.p_crossover).unwrap();
                let mutate = Bernoulli::new(self.p_mutate).unwrap();
                let mutate_value = Uniform::new_inclusive(u8::MIN, u8::MAX);
                for i in 0..parents[0].activations().len() {
                    let mut act = parents[current_parent].activations()[i];
                    
                    if crossover.sample(&mut rng) {
                        current_parent = if current_parent == 0 {1} else {0};
                    }

                    if mutate.sample(&mut rng) {
                        act = mutate_value.sample(&mut rng);
                    }

                    activations.push(act);
                }
            }
            
            (BinaryBrain::with_parameters(weights, activations, parents[0].input_count(), parents[0].output_count()).unwrap(), f64::MIN)
        }).take(pop_count).collect();
    }

    pub fn clone_fittest(&self) -> (BinaryBrain, f64) {
        let target = self.population.iter().max_by(
            |a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal)
        ).unwrap();

        (*target).clone()
    }
}