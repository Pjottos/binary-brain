use std::mem;
#[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
use core::arch::x86_64::*;

#[derive(Clone)]
pub struct Xoshiro128PlusPlusAvx2 {
    state: [[u64; Self::UNROLL]; 2]
}

impl Xoshiro128PlusPlusAvx2 {
    const UNROLL: usize = 4;

    #[inline]
    pub fn new(state: [u8; 8 * Self::UNROLL * 2]) -> Xoshiro128PlusPlusAvx2 {
        unsafe { Xoshiro128PlusPlusAvx2 {
            state: mem::transmute(state),
        } }
    }

    /// Generates 256 bits
    #[inline]
    pub fn next(&mut self) -> [u64; Self::UNROLL] {
        let mut s1: [u64; Self::UNROLL] = unsafe { mem::MaybeUninit::uninit().assume_init() };
        let mut result: [u64; Self::UNROLL] = unsafe { mem::MaybeUninit::uninit().assume_init() };

        (0..Self::UNROLL).for_each(|i| result[i] = (self.state[0][i] + self.state[1][i]).rotate_left(17) + self.state[0][i]);
        (0..Self::UNROLL).for_each(|i| s1[i] = self.state[0][i] ^ self.state[1][i]);
        (0..Self::UNROLL).for_each(|i| self.state[0][i] = self.state[0][i].rotate_left(49) ^ s1[i] ^ (s1[i] << 21));
        (0..Self::UNROLL).for_each(|i| self.state[1][i] = s1[i].rotate_left(28));

        result
    }

    /// Generates 256 bits where each bit has probability p/256 of being set.  
    /// Note that this is considerably slower than just calling `next` if you need a probability of ~0.5.
    #[inline]
    pub fn next_with_bias(&mut self, p: u8) -> [u64; Self::UNROLL] {
        #[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
        unsafe {
            let mut result = _mm256_setzero_si256();
            let treshold = _mm256_set1_epi8((127 - p) as i8);
            
            for _ in 0..8 {
                // shift the bits in each segment by 1 to make space for a new random bit
                result = _mm256_add_epi8(result, result);
                // generate uniform random bits
                let random: __m256i = mem::transmute(self.next());
                // compare random bits to treshold, each random segment has p/256 chance
                // of being greater than the treshold. therefore each segment in `compare`
                // has p/256 chance of being 0xFF or -1 since it's signed
                let compare = _mm256_cmpgt_epi8(random, treshold);
                // if a segment in `compare` is -1, this will set the lowest bit in `result`
                // otherwise it is left at 0
                result = _mm256_sub_epi8(result, compare);
            }

            return mem::transmute(result);
        }

        #[cfg(not(all(target_feature = "avx2", target_arch = "x86_64")))]
        {
            let mut result = [0i8; Self::UNROLL * 8];
            let treshold = [(127 - p) as i8; Self::UNROLL * 8];
            
            for _ in 0..8 {
                (0..Self::UNROLL * 8).for_each(|i| result[i] += result[i]);
                
                let random: [i8; Self::UNROLL * 8] = unsafe { mem::transmute(self.next()) };
                let mut compare: [i8; Self::UNROLL * 8] = unsafe { mem::MaybeUninit::uninit().assume_init() };
                (0..Self::UNROLL * 8).for_each(|i| compare[i] = if random[i] > treshold[i] {-1} else {0});
                    
                (0..Self::UNROLL * 8).for_each(|i| result[i] -= random[i]);
            }

            return unsafe { mem::transmute(result) };
        }
    }

    #[inline]
    pub fn jump(&mut self) {
        let jump_values = [0x2bd7a6a6e99c2ddc, 0x0992ccaf6a6fca05];

        for i in 0..Self::UNROLL {
            let mut s0 = 0;
            let mut s1 = 0;
            for jump in jump_values.iter() {
                for b in 0..64 {
                    if (jump & 1u64) << b != 0 {
                        s0 ^= self.state[0][i];
                        s1 ^= self.state[1][i];
                    }
                    self.next();	
                }
            }
            self.state[0][i] = s0;
            self.state[1][i] = s1;
        }
    }
}

// #[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
// #[inline]
// pub unsafe fn mm256_cmpge_epu8(a: __m256i, b: __m256i) -> __m256i {
//     _mm256_cmpeq_epi8(_mm256_max_epu8(a, b), a)
// }

// #[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
// #[inline]
// pub unsafe fn mm256_cmple_epu8(a: __m256i, b: __m256i) -> __m256i {
//     mm256_cmpge_epu8(b, a)
// }

// #[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
// #[inline]
// pub unsafe fn mm256_cmpgt_epu8(a: __m256i, b: __m256i) -> __m256i {
//     _mm256_xor_si256(mm256_cmple_epu8(a, b), _mm256_set1_epi8(-1))
// }

// #[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
// #[inline]
// pub unsafe fn mm256_cmplt_epu8(a: __m256i, b: __m256i) -> __m256i {
//     mm256_cmpgt_epu8(b, a)
// }


#[cfg(test)]
mod benches {
    use test::{Bencher, black_box};
    use super::*;
    use rand::prelude::*;

    #[bench]
    fn xorshiro128plusplus_next(b: &mut Bencher) {
        let mut state = [0u8; 64];
        thread_rng().fill_bytes(&mut state);
        let mut rng = Xoshiro128PlusPlusAvx2::new(state);

        b.iter(|| {
            rng.next()
        });
    }

    #[bench]
    fn xorshiro128plusplus_next_with_bias(b: &mut Bencher) {
        let mut state = [0u8; 64];
        thread_rng().fill_bytes(&mut state);
        let mut rng = Xoshiro128PlusPlusAvx2::new(state);

        b.iter(|| {
            let bias = black_box(7);
            rng.next_with_bias(bias);
        });
    }
}