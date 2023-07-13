use actix_web::body::BoxBody;
use actix_web::{
    get, http::header::ContentType, post, web, App, HttpResponse, HttpServer, Responder,
};
use discrete_logarithm_lib::{is_prime, shanks_algorithm_with_output, FpUnitsDiscLogSolver};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[derive(Deserialize)]
struct Inputs {
    prime: u32,
    base: u32,
    num: u32,
}

#[derive(Serialize)]
struct ShanksOutput {
    prime: u32,
    base: u32,
    num: u32,
    n: u32,
    log: u32,
    base_inverse: u32,
    collision_list1: HashMap<u32, i32>,
    collision_list2: HashMap<u32, i32>,
}

impl Responder for ShanksOutput {
    type Body = BoxBody;

    fn respond_to(self, req: &actix_web::HttpRequest) -> HttpResponse<Self::Body> {
        let body = serde_json::to_string(&self).unwrap();
        HttpResponse::Ok()
            .content_type(ContentType::json())
            .body(body)
    }
}
// struct AppSolver {
//     solver: Mutex<Option<FpUnitsDiscLogSolver>>,
// }

// impl AppSolver {
//     fn new() -> Self {
//         AppSolver {
//             solver: Mutex::new(None),
//         }
//     }
// }

// #[post("/new/{prime}")]
// async fn new_solver(path: web::Path<u32>, data: web::Data<AppSolver>) -> impl Responder {
//     let mut guard = if let Ok(guard_option) = data.as_ref().solver.lock() {
//         guard_option
//     } else {
//         return HttpResponse::Conflict().finish();
//     };
//     *guard = Some(FpUnitsDiscLogSolver::new(path.into_inner()));
//     HttpResponse::Ok().body("created solver successfully\n")
// }

// #[get("/modulus")]
// async fn get_modulus(data: web::Data<AppSolver>) -> impl Responder {
//     let guard = if let Ok(guard_option) = data.as_ref().solver.lock() {
//         guard_option
//     } else {
//         return HttpResponse::Conflict().finish();
//     };

//     if let Some(ref fp) = *guard {
//         return HttpResponse::Ok().body(format!("prime modulus is {}\n", fp.prime));
//     }

//     HttpResponse::Conflict().body("no solver created yet\n")
// }

#[get("/solve")]
async fn solve_discrete_logarithm(data: web::Json<Inputs>) -> impl Responder {
    let (prime, base, num) = (data.prime, data.base, data.num);
    if !is_prime(prime) || base % prime == 0 || num % prime == 0 {
        return HttpResponse::Conflict().body("invalid input");
    }

    let (log, base_inverse, n, list1, list2) = shanks_algorithm_with_output(prime, base, num);
    let log = if let Some(x) = log { x } else { 0 };

    // TODO: Figure out how to return error and output from same function.
    ShanksOutput {
        prime,
        base,
        num,
        log,
        base_inverse,
        n,
        collision_list1: list1,
        collision_list2: list2,
    }
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {}
