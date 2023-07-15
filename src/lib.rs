use std::collections::HashMap;
use std::error::Error;

pub fn is_prime(p: u32) -> bool {
    for i in 2..=(p / 2) {
        if p % i == 0 {
            return false;
        }
    }
    true
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

pub struct FpUnitsDiscLogSolver {
    pub prime: u32,
}

impl FpUnitsDiscLogSolver {
    pub fn new(p: u32) -> Self {
        assert!(is_prime(p));
        FpUnitsDiscLogSolver { prime: p }
    }

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

    pub fn compute_inverse(&self, g: u32) -> u32 {
        assert!(g % self.prime != 0);
        self.fast_power(g, self.prime - 2)
    }

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

    // pub fn shanks_algorithm_with_lists
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
}
