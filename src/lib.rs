use std::collections::HashMap;
use std::error::Error;

/// Simple function that uses a brute force technique to check if `p` is prime or not
/// Returns true if `p` is prime, otherwise it returns false
pub fn is_prime(p: u32) -> bool {
    for i in 2..=(p / 2) {
        if p % i == 0 {
            return false;
        }
    }
    true
}

/// Function that uses the seives algorithm for computing all primes up to and including a given input `n`.
pub fn compute_primes(n: u32) -> Vec<u32> {
    let mut primes = vec![true; (n + 1) as usize];
    let mut curr = 2_u32;

    while curr <= n {
        if primes[curr as usize] {
            let multiples = (curr * 2..=curr * (n / curr)).step_by(curr as usize);
            for multiple in multiples {
                primes[multiple as usize] = false;
            }
        }
        curr += 1;
    }

    primes
        .into_iter()
        .enumerate()
        .filter(|(i, b)| *i != 0 && *i != 1 && *b)
        .map(|(i, _)| i as u32)
        .collect()
}

/// Simple function that uses brute force to attempt a factorization of `n`
/// returns a vector of (prime, exponent) where prime is the prime factor and exponent is the highest power of that prime that divides n
pub fn factor(mut n: u32) -> Vec<(u32, u32, u32)> {
    let potential_prime_factors = PrimeGenerator::new(n / 2);
    let mut result = vec![];
    for factor in potential_prime_factors {
        if factor > n {
            break;
        } else if n % factor == 0 {
            let mut exp = 0;
            while n % factor == 0 && n > 0 {
                n /= factor;
                exp += 1;
            }
            result.push((factor, exp, u32::pow(factor, exp)));
        }
    }
    result
}

/// Function that implements the fast powering algorithm. Takes `prime` a prime, `g` a base and `exp` an exponent that we would
/// like to raise `g` to the power of. The return value is `g` raised to `exp` modulo `prime`, the function panics if `prime` is not prime.
pub fn fast_power(prime: u32, mut g: u32, mut exp: u32) -> u32 {
    assert!(is_prime(prime));
    let mut result = 1;
    while exp > 0 {
        if exp % 2 == 1 {
            result *= g;
            result %= prime;
        }

        g *= g;
        g %= prime;
        exp /= 2;
    }
    result
}

/// Function that will compute the multiplicative inverse of a given integer `g` modulo `prime` where `prime` must be a prime.
/// The function will panic if `prime` is not prime or if `prime` divides `g`.
pub fn compute_inverse(prime: u32, g: u32) -> u32 {
    assert!(is_prime(prime) && g % prime != 0);
    fast_power(prime, g, prime - 2)
}

/// Function that will compute the order of an integer `g` in the group of units from the field Fp, where p is the prime `prime`.
/// The funciton will panic if `prime` is not prime or `prime` divides `g`.
pub fn compute_order(prime: u32, mut g: u32) -> u32 {
    assert!(is_prime(prime) && g % prime != 0);
    let base = g;
    let mut n = 1;
    while g != 1 {
        g *= base;
        g %= prime;
        n += 1;
    }
    n
}

/// Function will solve the discrete logarithm of `num` with base `g` in the group of units from the field Fp, where p is the prime supplied argument `prime`.
/// The funciton will panic if `prime` is not prime, `prime` divides `num` or `prime` divides `g`.
pub fn shanks_algorithm(prime: u32, g: u32, num: u32) -> Option<u32> {
    assert!(is_prime(prime) && g % prime != 0 && num % prime != 0);
    // First compute order of g
    let base_order = compute_order(prime, g);
    let n = f32::floor(f32::sqrt(base_order as f32)) as u32 + 1;
    let mut prod = 1;

    let mut list1 = HashMap::new();

    for i in 0..(n as i32) {
        list1.insert(prod, i);
        prod *= g;
        prod %= prime;
    }

    // insert final value
    list1.insert(prod, n as i32);

    // Now compute second list, we don't actually need to allocate any memory for this only set up variables
    let mut u = 1;
    let prod_inverse = compute_inverse(prime, prod);

    for i in 0..=(n as i32) {
        // Check if we have a match
        if let Some(x) = list1.get(&(num * u)) {
            let res = (x + (i * (n as i32))) as u32;
            return Some(res);
        }
        // otherwise update value of u
        u *= prod_inverse;
        u %= prime;
    }

    None
}

/// Function will solve the discrete logarithm of `num` with base `g` in the group of units from the field Fp, where p is the prime supplied argument `prime`.
/// The function will also return the lists tat were generated the collision. The funciton will panic if `prime` is not prime,
/// `prime` divides `num` or `prime` divides `g`.
pub fn shanks_algorithm_with_output(
    prime: u32,
    g: u32,
    num: u32,
) -> (Option<u32>, u32, u32, HashMap<u32, i32>, HashMap<u32, i32>) {
    assert!(is_prime(prime) && g % prime != 0 && num % prime != 0);

    // Compute order of base
    let order = compute_order(prime, g);
    let n = f32::floor(f32::sqrt(order as f32)) as u32 + 1;

    // Generate our first list
    let mut list1 = HashMap::new();
    let mut prod = 1;

    for i in 0..(n as i32) {
        list1.insert(prod, i);
        prod *= g;
        prod %= prime;
    }

    // insert final value into the list1
    list1.insert(g, n as i32);

    // Now compute the inverse of base^n, note g = base^n from our previous computation
    let mut list2 = HashMap::new();
    let mut u = 1;
    let base_inverse = compute_inverse(prime, prod);

    for i in 0..=(n as i32) {
        // First always insert into list2
        list2.insert(num * u, i);
        // Then check if we have a match in list1
        if let Some(x) = list1.get(&((num * u) % prime)) {
            let res = x + (i * (n as i32));
            return (Some(res as u32), base_inverse, n, list1, list2);
        }

        // otherwise update u and proceed
        u *= base_inverse;
        u %= prime;
    }

    // there is no discrete logarithm for the given base and num
    (None, base_inverse, n, list1, list2)
}

/// A struct that will generate prime numbers that will generate prime numbers up to and including some final number.
pub struct PrimeGenerator {
    n: u32,
    primes: Vec<bool>,
    curr: u32,
}

impl PrimeGenerator {
    /// Creates a new `PrimeGenerator`
    pub fn new(n: u32) -> Self {
        let primes = vec![true; (n + 1) as usize];
        let curr = 2_u32;
        PrimeGenerator { n, primes, curr }
    }
}

impl std::iter::Iterator for PrimeGenerator {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        while self.curr <= self.n {
            if self.primes[self.curr as usize] {
                let multiples =
                    (self.curr * 2..=self.curr * (self.n / self.curr)).step_by(self.curr as usize);
                for multiple in multiples {
                    self.primes[multiple as usize] = false;
                }
                let res = self.curr;
                self.curr += 1;
                return Some(res);
            } else {
                self.curr += 1;
            }
        }
        None
    }
}

/// A struct that encapsulates all the necessary functions for solving the discrete logarithm in the group of units from the field Fp.
/// A `FpUnitsDiscLogSolver` may be preferable in many cases since only one check is required at initialization to ensure the modulus is prime,
/// where as, many repeated checks that the modulus is prime will be needed if using only the functions declared in this module
pub struct FpUnitsDiscLogSolver {
    pub prime: u32,
}

impl FpUnitsDiscLogSolver {
    /// Create a new `FpUnitsDiscLogSolver`, will panic if `p` is not prime.
    pub fn new(p: u32) -> Self {
        assert!(is_prime(p));
        FpUnitsDiscLogSolver { prime: p }
    }

    /// Performs the fast power algorithm for computing a base raised to an exponenet in the group of units from Fp.
    /// In this method the parameter `g` is the base and `exp` is the power we wish to raise `g` to.
    /// The method panics if `g` % self.prime == 0, since in this case `g` is not in the group of units to begin with.
    pub fn fast_power(&self, mut g: u32, mut exp: u32) -> u32 {
        assert!(g % self.prime != 0);
        let mut result = 1;
        // let base = g;
        while exp > 0 {
            if exp % 2 == 1 {
                result *= g;
                result %= self.prime;
            }
            g *= g;
            g %= self.prime;
            exp /= 2;
        }
        result
    }

    /// Computes the inverse of a given element `g` from the group of units in Fp.
    /// Method will panic if `g` % self.prime == 0, since in this case `g` is not in the group of units to begin with.
    pub fn compute_inverse(&self, g: u32) -> u32 {
        assert!(g % self.prime != 0);
        self.fast_power(g, self.prime - 2)
    }

    /// Computes the order of a given element `g` from the group of units in Fp.
    /// Method will panic if `g` % self.prime == 0, since in this case `g` is not in the group of units to begin with.
    pub fn compute_order(&self, g: u32) -> u32 {
        assert!(g % self.prime != 0);
        let mut g = g;
        let base = g;
        let mut n: u32 = 1;
        while g != 1 {
            g *= base;
            g %= self.prime;
            n += 1;
        }
        n
    }

    /// Attempts to solve the discrete logarithm problem. `num` is the value whose logarithm we are trying to find with the given `base`.
    /// If the discrete logarithm exists, a `Some(u32)` will be returned, otherwise `None` will be returned.
    /// Method will panic if `base` % self.prime == 0 or `num` % self.prime == 0, since in this case either `base` or `num` is not in the group of units to begin with.
    pub fn shanks_algorithm(&self, base: u32, num: u32) -> Option<u32> {
        assert!(base % self.prime != 0 && num % self.prime != 0);
        // First compute order of base
        let order = self.compute_order(base);
        let n = f32::floor(f32::sqrt(order as f32)) as u32 + 1;

        // Generate our first list
        let mut list1 = HashMap::new();
        let mut g = 1;

        for i in 0..(n as i32) {
            list1.insert(g, i);
            g *= base;
            g %= self.prime;
        }

        // Insert final value for base and exponent into first list
        list1.insert(g, n as i32);

        let mut u = 1;
        let base_inverse = self.compute_inverse(g);

        for i in 0..=(n as i32) {
            // Check if we have a match
            if let Some(x) = list1.get(&((num * u) % self.prime)) {
                let res = x + (i * (n as i32));
                return Some(res as u32);
            }
            // Otherwise update value of u
            u *= base_inverse;
            u %= self.prime;
        }
        None
    }

    /// Attempts to solve the discrete logarithm problem. `num` is the value whose logarithm we are trying to find with the given `base`.
    /// This method also returns the data that may be of interest to some that was generated during the algorithm in an attempt to find the logarithm, eg the list of elements that were generated when trying to find a collision, etc...
    /// If the discrete logarithm exists, a `Some(u32)` will be returned, otherwise `None` will be returned in the tuple containing all of the outputs.
    /// Method will panic if `base` % self.prime == 0 or `num` % self.prime == 0, since in this case either `base` or `num` is not in the group of units to begin with.
    pub fn shanks_algorithm_with_output(
        &self,
        base: u32,
        num: u32,
    ) -> (Option<u32>, u32, u32, HashMap<u32, i32>, HashMap<u32, i32>) {
        assert!(base % self.prime != 0 && num % self.prime != 0);

        // Compute order of base
        let order = self.compute_order(base);
        let n = f32::floor(f32::sqrt(order as f32)) as u32 + 1;

        // Generate our first list
        let mut list1 = HashMap::new();
        let mut g = 1;

        for i in 0..(n as i32) {
            list1.insert(g, i);
            g *= base;
            g %= self.prime;
        }

        // insert final value into the list1
        list1.insert(g, n as i32);

        // Now compute the inverse of base^n, note g = base^n from our previous computation
        let mut list2 = HashMap::new();
        let mut u = 1;
        let base_inverse = self.compute_inverse(g);

        for i in 0..=(n as i32) {
            // First always insert into list2
            list2.insert(num * u, i);
            // Then check if we have a match in list1
            if let Some(x) = list1.get(&((num * u) % self.prime)) {
                let res = x + (i * (n as i32));
                return (Some(res as u32), u, n, list1, list2);
            }

            // otherwise update u and proceed
            u *= base_inverse;
            u %= self.prime;
        }

        // there is no discrete logarithm for the given base and num
        (None, u, n, list1, list2)
    }

    /// A method that will solve the discrete logarithm where the modulus is a prime power.
    /// Can be called as a stand alone function or as a helper method.
    pub fn solve_prime_power(&self, base: u32, num: u32, prime: u32, exp: u32) -> Option<u32> {
        let mut x = 0;
        let mut u = 1;
        let mut v = u32::pow(prime, exp - 1);
        // let b = u32::pow(base, u32::pow(prime, v));
        let b = self.fast_power(base, v);

        for _i in 0..exp {
            let curr_inverse = self.compute_inverse(self.fast_power(base, x));
            if let Some(x_i) =
                self.shanks_algorithm(b, self.fast_power((num * curr_inverse) % self.prime, v))
            {
                x += x_i * u;
                x %= u32::pow(prime, exp);
                u *= prime;
                v /= prime;
            } else {
                return None;
            }
        }

        Some(x)
    }

    /// A method for computing the discrete logarithm using Pollhig-Hellman algorithm.
    /// Method will panic if `base` % self.prime == 0 or `num` % self.prime == 0, since in this case either `base` or `num` is not in the group of units to begin with.
    pub fn pollhig_hellman(&self, base: u32, num: u32) -> Option<u32> {
        assert!(base % self.prime != 0 && num % self.prime != 0);

        // First compute order of base
        let order = self.compute_order(base);

        // Compute prime powers of order
        let prime_powers = factor(order);
        let mut solutions = vec![];

        for (prime, power, prime_power) in prime_powers {
            solutions.push((
                self.solve_prime_power(
                    u32::pow(base, order / prime_power),
                    u32::pow(num, order / prime_power),
                    prime,
                    power,
                ),
                prime_power,
            ));
        }

        // Use chinese remainder theorm to solve the simulataneous system of congruents
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_inverse() {
        let solver = FpUnitsDiscLogSolver::new(71);
        let base = 11;
        let inverse = solver.compute_inverse(11);

        println!("g = {}, g inverse = {}", base, inverse);
        println!("g * g^-1 = {}", (base * inverse) % 71);

        assert_eq!((base * inverse) % 71, 1);

        let solver = FpUnitsDiscLogSolver::new(17389);
        let base = 9704;
        let inverse = solver.compute_inverse(9704);

        println!("g = {}, g inverse = {}", base, inverse);
        println!("g * g^-1 = {}", (base * inverse) % 17389);

        assert_eq!((base * inverse) % 17389, 1);
    }

    #[test]
    fn test_compute_order() {
        let solver = FpUnitsDiscLogSolver::new(71);
        let base = 11;
        let order: u32 = solver.compute_order(base);
        println!("g = {}, order of g = {}", base, order);
        println!("g^n = {}", solver.fast_power(base, order));

        assert_eq!(solver.fast_power(base, order), 1);

        let solver = FpUnitsDiscLogSolver::new(17389);
        let base = 9704;
        let order: u32 = solver.compute_order(base);
        println!("g = {}, order of g = {}", base, order);
        println!("g^n = {}", solver.fast_power(base, order));

        assert_eq!(solver.fast_power(base, order), 1);
    }

    #[test]
    fn test_shanks_algorithm() {
        let solver = FpUnitsDiscLogSolver::new(71);
        let base = 11;
        if let Some(x) = solver.shanks_algorithm(base, 23) {
            println!("{}^{} = {}", base, x, solver.fast_power(base, x));
            assert_eq!(solver.fast_power(base, x), 23);
        } else {
            panic!();
        }

        let solver = FpUnitsDiscLogSolver::new(17389);
        let base = 9704;
        if let Some(x) = solver.shanks_algorithm(base, 13896) {
            println!("{}^{} = {}", base, x, solver.fast_power(base, x));
            assert_eq!(solver.fast_power(base, x), 13896);
        }

        assert!(true);
    }

    #[test]
    fn test_shanks_algorithm_with_output() {
        let solver = FpUnitsDiscLogSolver::new(71);
        let base = 11;
        match solver.shanks_algorithm_with_output(base, 23) {
            (Some(x), _, _, l1, l2) => {
                println!("list1: {:#?}", l1);
                println!("list2: {:#?}", l2);
                println!("log = {}", x);
                println!("{}^{} = {}", base, x, solver.fast_power(base, x));

                assert_eq!(solver.fast_power(base, x), 23);
            }
            (None, _, _, _l1, _l2) => panic!(),
            _ => panic!(),
        }

        let solver = FpUnitsDiscLogSolver::new(17389);
        let base = 9704;

        match solver.shanks_algorithm_with_output(base, 13896) {
            (Some(x), _, _, l1, l2) => {
                println!("list1: {:#?}", l1);
                println!("list2: {:#?}", l2);
                println!("log = {}", x);
                println!("{}^{} = {}", base, x, solver.fast_power(base, x));

                assert_eq!(solver.fast_power(base, x), 13896);
            }
            (None, _, _, _l1, _l2) => panic!(),
            _ => panic!(),
        }

        assert!(true);
    }

    #[test]
    fn test_compute_primes() {
        let prime = 31_u32;
        assert_eq!(
            vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31],
            compute_primes(prime)
        );
        assert_eq!(
            vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31],
            compute_primes(32)
        );

        println!("{:?}", compute_primes(32));
        println!("{:?}", compute_primes(1000));

        // tests passed
        assert!(true);
    }

    #[test]
    fn test_prime_generator() {
        let prime_generator = PrimeGenerator::new(1000);

        assert_eq!(prime_generator.collect::<Vec<u32>>(), compute_primes(1000));
    }

    #[test]
    fn test_factor() {
        let N = 2 * 2 * 28 * 3 * 7 * 15 * 9;
        println!("{N}");
        let factors = factor(N as u32);
        println!("{:?}", factors);

        assert_eq!(vec![(2, 4, 16), (3, 4, 81), (5, 1, 5), (7, 2, 49)], factors);
    }

    #[test]
    fn test_solve_prime_power() {
        let solver = FpUnitsDiscLogSolver::new(11251);
        if let Some(x) = solver.solve_prime_power(5448, 6909, 5, 4) {
            println!("{x}");
            assert!(x % 11251 == 511);
        } else {
            println!("panicing returned None");
            panic!();
        }
    }
}
