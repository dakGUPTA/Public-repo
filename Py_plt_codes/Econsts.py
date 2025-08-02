mu_m = mu_p = 440
nu_m = nu_p = 0.3
Az_m = Az_p = 3.0


C11m = (mu_m*(2.0 * (2.0 + Az_m) / (1.0 + Az_m)-
        (1.0 - 4.0 * nu_m) / (1.0 - 2.0 * nu_m)));

C12m = (mu_m*(2.0 * (Az_m / (1.0 + Az_m)) -
        (1.0 - 4.0 * nu_m) / (1.0 - 2.0 * nu_m)));

C44m = mu_m * (2.0 * Az_m / (1.0 + Az_m));

C11p = (mu_p*(2.0*(2.0 + Az_p)/(1.0 + Az_p)-
        (1.0 - 4.0 * nu_p)/(1.0 - 2.0 * nu_p)));

C12p = (mu_p* (2.0*(Az_p/(1.0+Az_p))-
        (1.0 - 4.0 * nu_p) / (1.0 - 2.0 * nu_p)));

C44p = mu_p * (2.0*Az_p /(1.0 + Az_p));

print(f"C11m, C12m, C44m : {C11m}, {C12m}, {C44m}")
print(f"C11p, C12p, C44p : {C11p}, {C12p}, {C44p}")
