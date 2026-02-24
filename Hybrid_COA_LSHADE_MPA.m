% % Hybrid_V15: Clean, Efficient Hybrid with Current-to-Best and Top-Pbest Guidance
% 
function [best_solution, best_fitness, curve_f, global_Cov, exec_time_ms, iter_found] = Hybrid_COA_LSHADE_MPA(N, T, lb, ub, dim, fobj)
    % Allow CEC2019 and CEC2022 dimensions
valid_dims_cec2019 = [9, 10, 16, 18];
valid_dims_cec2022 = [2, 10, 20];
valid_dims_real = [2,3, 4, 5];
valid_dims_math = [2,3, 4, 5,6,10,30];
valid_dims = unique([valid_dims_cec2019, valid_dims_cec2022,valid_dims_real,valid_dims_math]);

if ~ismember(dim, valid_dims)
    error('Unsupported problem dimension: %d. Only CEC2019 and CEC2022 dimensions are supported.', dim);
end


    tic;
    population = lb + (ub - lb) .* rand(N, dim);
    fitness = arrayfun(@(i) fobj(population(i, :)), 1:N);
    [best_fitness, best_idx] = min(fitness);
    best_solution = population(best_idx, :);

    F = 0.5 + 0.3 * rand(N, 1);
    CR = 0.9 * ones(N, 1);
    curve_f = zeros(T, 1);
    global_Cov = zeros(T, 1);
    max_successful_entries = N * T;
    successful_F = NaN(max_successful_entries, 1);
    successful_CR = NaN(max_successful_entries, 1);
    success_count = 0;
    iter_found = NaN;
    elite_memory = population;
    stagnation_counter = 0;

    for t = 1:T
        new_pop = population;
        new_fit = fitness;
        [~, sort_idx] = sort(fitness);
        elite_memory = population(sort_idx(1:ceil(0.2 * N)), :);

        for i = 1:N
            % Current-to-best/1 mutation with p-best from top 20%
            pbest = elite_memory(randi(size(elite_memory, 1)), :);
            r1 = randi(N); while r1 == i, r1 = randi(N); end
            r2 = randi(N); while any([r2 == i, r2 == r1]), r2 = randi(N); end

            donor = population(i,:) + F(i) * (pbest - population(i,:)) + F(i) * (population(r1,:) - population(r2,:));
            donor = bound_check(donor, lb, ub);
            trial = crossover_LSHADE(population(i,:), donor, CR(i), dim);
            trial = bound_check(trial, lb, ub);

            trial_fit = fobj(trial);
            if trial_fit < fitness(i)
                new_pop(i,:) = trial;
                new_fit(i) = trial_fit;
                success_count = success_count + 1;
                successful_F(success_count) = F(i);
                successful_CR(success_count) = CR(i);

                if trial_fit < best_fitness
                    best_fitness = trial_fit;
                    best_solution = trial;
                    iter_found = t;
                    stagnation_counter = 0;
                end
            end
        end

        population = new_pop;
        fitness = new_fit;

        stagnation_counter = stagnation_counter + 1;
        if stagnation_counter > 20 && mod(t, 10) == 0
            inject = randperm(N, ceil(0.1*N));
            population(inject,:) = lb + (ub - lb) .* rand(length(inject), dim);
            stagnation_counter = 0;
        end

        curve_f(t) = best_fitness;
        global_Cov(t) = mean(fitness);
    end

    exec_time_ms = toc ; %it is in seconds not ms
end

function vec = bound_check(vec, lb, ub)
    vec = min(max(vec, lb), ub);
end

function trial = crossover_LSHADE(target, donor, CR, dim)
    trial = target;
    jrand = randi(dim);
    for j = 1:dim
        if rand() < CR || j == jrand
            trial(j) = donor(j);
        end
    end
end



