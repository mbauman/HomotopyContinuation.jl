export monodromy_solve, find_start_pair
#
# MonodromyResult,
# real_solutions,
# is_success,
# is_heuristic_stop,
# nreal,
# parameters,
# verify_solution_completeness,
# solution_completeness_witnesses,
# permutations


#####################
# Monodromy Options #
#####################


## Options
"""
    MonodromyOptions(; options...)

Options for [`monodromy_solve`](@ref).

## Options

* `check_startsolutions=true`: If `true`, we do a Newton step for each entry of `sols`for
  checking if it is a valid startsolutions. Solutions which are not valid are sorted out.
* `distance = EuclideanNorm()`: The distance function used for [`UniquePoints`](@ref).
* `loop_finished_callback=always_false`: A callback to end the computation. This function is
  called with all current [`PathResult`](@ref)s after a loop is exhausted.
  2 arguments. Return `true` if the compuation should be stopped.
* `equivalence_classes=true`: This only applies if there is at least one group action
  supplied. We then consider two solutions in the same equivalence class if we can transform
  one to the other by the supplied group actions. We only track one solution per equivalence
  class.
* `group_action=nothing`: A function taking one solution and returning other solutions if
  there is a constructive way to obtain them, e.g. by symmetry.
* `group_actions=nothing`: If there is more than one group action you can use this to chain
  the application of them. For example if you have two group actions `foo` and `bar` you can
  set `group_actions=[foo, bar]`. See [`GroupActions`](@ref) for details regarding the
  application rules.
* `max_loops_no_progress::Int=10`: The maximal number of iterations (i.e. loops generated)
  without any progress.
* `min_solutions`: The minimal number of solutions before a stopping heuristic is applied.
  By default no lower limit is enforced.
* `parameter_sampler = independent_normal`: A function taking the parameter `p` and
  returning a new random parameter `q`. By default each entry of the parameter vector
  is drawn independently from Normal distribution.
* `resuse_loops::Symbol=:all`: Strategy to reuse other loops for new found solutions. `:all`
  propagates a new solution through all other loops, `:random` picks a random loop, `:none`
  doesn't reuse a loop.
* `target_solutions_count`: The computation is stopped if this number of solutions is
  reached.
* `threading = true`: Enable multithreading of the path tracking.
* `timeout`: The maximal number of *seconds* the computation is allowed to run.
"""
Base.@kwdef struct MonodromyOptions{D,F1,GA<:Union{Nothing,GroupActions},F2}
    check_startsolutions::Bool = true
    distance::D = EuclideanNorm()
    group_actions::GA = nothing
    loop_finished_callback::F1 = always_false
    parameter_sampler::F2 = independent_normal
    equivalence_classes::Bool = !isnothing(group_actions)
    # stopping heuristic
    target_solutions_count::Union{Nothing,Int} = nothing
    timeout::Union{Nothing,Float64} = nothing
    min_solutions::Union{Nothing,Int} = nothing
    max_loops_no_progress::Int = 10
    reuse_loops::Symbol = :all
end

"""
    independent_normal(p::AbstractVector)

Sample a vector where each entry is drawn independently from the complex Normal distribution
by calling [`randn(ComplexF64)`](@ref).
"""
independent_normal(p::AbstractVector{T}) where {T} = randn(ComplexF64, length(p))

##########################
## Monodromy Statistics ##
##########################

Base.@kwdef mutable struct MonodromyStatistics
    tracked_loops::Threads.Atomic{Int} = Threads.Atomic{Int}(0)
    tracking_failures::Threads.Atomic{Int} = Threads.Atomic{Int}(0)
    loops::Int = 1
    # nreal::Threads.Atomic{Int}
    # nparametergenerations::Threads.Atomic{Int}
    solutions_development::Vector{Int} = Int[]
    permutations::Vector{Dict{Int,Int}} = Dict{Int,Int}[]
end

MonodromyStatistics(nsolutions::Int) =
    MonodromyStatistics(solutions_development = [nsolutions])


function Base.show(io::IO, S::MonodromyStatistics)
    println(io, "MonodromyStatistics")
    print_fieldname(io, S, :tracked_loops)
    print_fieldname(io, S, :tracking_failures)
    print_fieldname(io, S, :loops)
    print_fieldname(io, S, :solutions_development)
end
Base.show(io::IO, ::MIME"application/prs.juno.inline", S::MonodromyStatistics) = S

# update routines
function trackedloop!(stats::MonodromyStatistics, retcode)
    if is_success(retcode)
        Threads.atomic_add!(stats.tracked_loops, 1)
    else
        Threads.atomic_add!(stats.tracking_failures, 1)
    end
    stats
end

function finished!(stats, nsolutions)
    push!(stats.nsolutions_development, nsolutions)
end

function n_completed_loops_without_change(stats, nsolutions)
    k = 0
    for i = length(stats.nsolutions_development):-1:1
        if stats.nsolutions_development[i] != nsolutions
            return max(k - 1, 0)
        end
        k += 1
    end
    return max(k - 1, 0)
end

function n_solutions_current_loop(statistics, nsolutions)
    nsolutions - statistics.nsolutions_development[end]
end

#############################
# Loops and Data Structures #
#############################


#######################
# Loop data structure
#######################


struct MonodromyLoop
    # p -> p₁ -> p₂ -> p
    p::Vector{ComplexF64}
    p₁::Vector{ComplexF64}
    p₂::Vector{ComplexF64}
end

function MonodromyLoop(base, options::MonodromyOptions)
    p = convert(Vector{ComplexF64}, base)
    p₁ = convert(Vector{ComplexF64}, options.parameter_sampler(p))
    p₂ = convert(Vector{ComplexF64}, options.parameter_sampler(p))

    MonodromyLoop(p, p₁, p₂)
end


vec_angle(p, q) = acos(real(LA.dot(p,q))/ (LA.norm(p) * LA.norm(q)))



#####################
## monodromy solve ##
#####################

mutable struct MonodromySolver{
    Tracker<:EndgameTracker,
    UP<:UniquePoints,
    MO<:MonodromyOptions,
}
    trackers::Vector{Tracker}
    loops::Vector{MonodromyLoop}
    unique_points::UP
    unique_points_lock::ReentrantLock
    options::MO
    statistics::MonodromyStatistics
end

function MonodromySolver(
    F::Union{System,AbstractSystem};
    tracker_options = TrackerOptions(),
    options = MonodromyOptions(),
)
    if !isa(options, MonodromyOptions)
        options = MonodromyOptions(; options...)
    end

    egtracker = parameter_homotopy(
        F;
        generic_parameters = randn(ComplexF64, nparameters(F)),
        tracker_options = tracker_options,
        endgame_options = EndgameOptions(endgame_start = 0.0),
    )
    trackers = [egtracker]
    group_actions = options.equivalence_classes ? options.group_actions : nothing
    x₀ = zeros(ComplexF64, size(F, 2))
    unique_points =
        UniquePoints(x₀, 1; metric = options.distance, group_actions = group_actions)
    statistics = MonodromyStatistics()

    MonodromySolver(
        trackers,
        MonodromyLoop[],
        unique_points,
        ReentrantLock(),
        options,
        MonodromyStatistics(),
    )
end

add_loop!(MS::MonodromySolver, p) = push!(MS.loops, MonodromyLoop(p, MS.options))
loop(MS::MonodromySolver, i) = MS.loops[i]
nloops(MS::MonodromySolver) = length(MS.loops)

"""
    monodromy_solve(F, [sols, p]; options = MonodromyOptions(), tracker_options = TrackerOptions())

Solve a polynomial system `F(x;p)` with specified parameters and initial solutions `sols`
by monodromy techniques. This makes loops in the parameter space of `F` to find new solutions.
If `F` the parameters `p` only occur *linearly* in `F` it is eventually possible to compute
a *start pair* ``(x₀, p₀)`` automatically. In this case `sols` and `p` can be omitted and
the automatically generated parameters can be obtained with the [`parameters`](@ref) function
from the [`MonodromyResult`](@ref).
See also [`MonodromyOptions`](@ref).
"""
function monodromy_solve(F::Vector{<:ModelKit.Expression}, args...; parameters, kwargs...)
    monodromy_solve(System(F; parameters = parameters), args...; kwargs...)
end
function monodromy_solve(
    F::AbstractVector{<:MP.AbstractPolynomial},
    args...;
    parameters,
    variables = setdiff(MP.variables(F), parameters),
)
    sys = System(F, variables = variables, parameters = parameters)
    monodromy_solve(sys, args...; kwargs...)
end
function monodromy_solve(
    F::Union{System,AbstractSystem},
    args...;
    seed = rand(UInt32),
    options = MonodromyOptions(),
    tracker_options = TrackerOptions(),
    kwargs...,
)
    if !isnothing(seed)
        Random.seed!(seed)
    end
    MS = MonodromySolver(F; options = options, tracker_options = tracker_options)
    if length(args) == 0
        start_pair = find_start_pair(F)
        if isnothing(start_pair)
            error(
                "Cannot compute a start pair (x, p) using `find_start_pair(F)`." *
                " You need to explicitly pass a start pair.",
            )
        end
        x, p = start_pair
        monodromy_solve(MS, [x], p, seed; kwargs...)
    else
        monodromy_solve(MS, args..., seed; kwargs...)
    end
end

"""
    find_start_pair(F; max_tries = 100, atol = 0.0, rtol = 1e-12)

Try to find a pair `(x,p)` for the system `F` such that `F(x,p) = 0` by randomly sampling
a pair `(x₀, p₀)` and performing Newton's method in variable *and* parameter space.
"""
find_start_pair(F::System; kwargs...) = find_start_pair(ModelKitSystem(F); kwargs...)
function find_start_pair(
    F::AbstractSystem;
    max_tries::Int = 100,
    atol::Float64 = 0.0,
    rtol::Float64 = 1e-12,
)
    SF = StartPairSystem(F)
    m = nparameters(F)
    n = nvariables(F)
    c = NewtonCache(SF)
    for i = 1:max_tries
        xp₀ = randn(ComplexF64, size(SF, 2))
        res = newton(SF, xp₀, nothing, InfNorm(), c, atol = atol, rtol = rtol)
        if is_success(res)
            x, p = res.x[1:n], res.x[n+1:end]
            res2 = newton(F, x, p, InfNorm())
            if is_success(res2)
                return res2.x, p
            end
        end
    end
    nothing
end

monodromy_solve(MS::MonodromySolver, x::AbstractVector{<:Number}, p, seed; kwargs...) =
    monodromy_solve(MS, [x], p, seed; kwargs...)

function monodromy_solve(
    MS::MonodromySolver,
    X::AbstractVector{<:AbstractVector{<:Number}},
    p,
    seed;
    show_progress::Bool = true,
    threading::Bool = Threads.nthreads() > 1,
)
    if !show_progress
        progress = nothing
    else
        if MS.options.equivalence_classes
            desc = "Solutions (modulo group action) found:"
        else
            desc = "Solutions found:"
        end
        progress = ProgressMeter.ProgressUnknown(; dt = 0.3, desc = desc)
        progress.tlast += 0.5
    end
    results = check_start_solutions(MS, X, p)

    if isempty(results)
        @warn "None of the provided solutions is a valid start solution."
        return results
    end

    # if threading
    serial_monodromy_solve!(results, MS, p, seed, progress), p, MS.loops
    # else
    #     threaded_monodromy_solve!(results, MS, p, seed, progress)
    # end
end

function check_start_solutions(MS::MonodromySolver, X, p)
    tracker = MS.trackers[1]
    start_parameters!(tracker, p)
    target_parameters!(tracker, p)
    results = PathResult[]
    for x in X
        res = track(tracker, x)
        if is_success(res)
            _, added = add!(MS, res, length(results) + 1)
            if added
                push!(results, res)
            end
        end
    end

    results
end

function add!(MS::MonodromySolver, res::PathResult, id)
    rtol = clamp(0.25 * inv(res.ω)^2, 1e-14, sqrt(res.accuracy))
    add!(
        MS.unique_points,
        solution(res),
        id;
        atol = 1e-14,
        rtol = rtol,
    )
end

struct LoopTrackingJob
    id::Int
    loop_id::Int
end


function serial_monodromy_solve!(
    results::Vector{PathResult},
    MS::MonodromySolver,
    p,
    seed,
    progress,
)::Vector{PathResult}
    queue = LoopTrackingJob[]
    tracker = MS.trackers[1]
    t₀ = time()
    @show length(results)
    nresults_loop_finish = length(results)
    loops_no_change = 0
    while true
        add_loop!(MS, p)

        # schedule all jobs
        new_loop_id = nloops(MS)
        for i = 1:length(results)
            push!(queue, LoopTrackingJob(i, new_loop_id))
        end

        while !isempty(queue)
            job = popfirst!(queue)
            res = track(tracker, solution(results[job.id]), loop(MS, job.loop_id))
            if !isnothing(res)
                # 1) check whether solutions already exists

                # we should have a region of uniqueness of 0.5inv(ω) * norm(x)
                # the norm(x) factors comes from the fact that we computed with a
                # weighted norm.
                # To be more pessimistic we only consider 0.25inv(ω)^2
                # and require at most sqrt(res.accuracy) and at least 1e-14.
                rtol = clamp(0.25 * inv(res.ω)^2, 1e-14, sqrt(res.accuracy))

                id, got_added = add!(
                    MS.unique_points,
                    solution(res),
                    length(results) + 1;
                    atol = 1e-14,
                    rtol = rtol,
                )
                got_added || continue
                # 2) doesn't exist, so add to results
                push!(results, res)

                # 3) schedule on same loop again
                push!(queue, LoopTrackingJob(id, job.loop_id))

                # 4) schedule on other loops
                if MS.options.reuse_loops == :all
                    for k = 1:nloops(MS)
                        k != job.loop_id || continue
                        push!(queue, LoopTrackingJob(id, k))
                    end
                elseif MS.options.reuse_loops == :random && nloops(MS) ≥ 2
                    k = rand(2:nloops(MS))
                    if k ≤ job.loop_id
                        k -= 1
                    end
                    push!(queue, LoopTrackingJob(id, k))
                end

                # 5) update progres
                if !isnothing(progress)
                    ProgressMeter.update!(progress, id)
                end
            else
                println("failed")
            end

            if length(results) == MS.options.target_solutions_count
                return results
            end
            if !isnothing(MS.options.timeout)
                if time() - t₀ > MS.options.timeout
                    return results
                end
            end
        end

        if length(results) == nresults_loop_finish
            loops_no_change += 1
        else
            nresults_loop_finish = length(results)
            loops_no_change = 0
        end
        @show length(results)

        if loops_no_change ≥ MS.options.max_loops_no_progress
            return results
        end
    end
end


"""
    track(tracker, x, edge::LoopEdge, loop::MonodromyLoop, stats::MonodromyStatistics)

Track `x` along the edge `edge` in the loop `loop` using `tracker`. Record statistics
in `stats`.
"""
function track(egtracker::EndgameTracker, x, loop::MonodromyLoop)
    tracker = egtracker.tracker

    start_parameters!(tracker, loop.p)
    target_parameters!(tracker, loop.p₁)
    retcode = track!(tracker, x)
    is_success(retcode) || return nothing

    start_parameters!(tracker, loop.p₁)
    target_parameters!(tracker, loop.p₂)
    retcode = track!(tracker, tracker.state.x)
    is_success(retcode) || return nothing

    start_parameters!(tracker, loop.p₂)
    target_parameters!(tracker, loop.p)
    result = track(egtracker, tracker.state.x)
    if !is_success(result)
        return nothing
    else
        result
    end
end







#
#
#
#
#
#
#
#
#
#
#
# """
#     MonodromyJob
#
# A `MonodromyJob` is consisting of a solution id and a loop id.
# """
#
# function monodromy_solve!(
#     MS::MonodromySolver,
#     seed;
#     show_progress::Bool = true,
#     threading::Bool = true,
# )
#
#     # add only unique points that are true solutions
#     for s in startsolutions
#         if options.check_startsolutions
#             init!(egtracker.tracker, s, 1.0, 0.0)
#             is_invalid_startvalue(egtracker.tracker.state.code)
#         else
#
#
#             add!(uniquepoints, s; tol = options.identical_tol)
#         end
#     end
#
#
#
#     if nsolutions(MS) == 0
#         @warn "None of the provided solutions is a valid start solution."
#         return MonodromyResult(
#             :invalid_startvalue,
#             similar(MS.solutions.points, 0),
#             MS.parameters,
#             MS.statistics,
#             MS.options.equivalence_classes,
#             seed,
#         )
#     end
#     # solve
#     retcode = :not_assigned
#     if show_progress
#         if !MS.options.equivalence_classes
#             desc = "Solutions found:"
#         else
#             desc = "Classes of solutions (modulo group action) found:"
#         end
#         progress =
#             ProgressMeter.ProgressUnknown(desc; delay = 0.5, clear_output_ijulia = true)
#     else
#         progress = nothing
#     end
#
#     n_blas_threads = single_thread_blas()
#
#     retcode = monodromy_solve!(MS, seed, progress; threading = threading)
#
#     n_blas_threads > 1 && set_num_BLAS_threads(n_blas_threads)
#     finished!(MS.statistics, nsolutions(MS))
#
#
#     MonodromyResult(
#         retcode,
#         solutions(MS),
#         MS.parameters,
#         MS.statistics,
#         MS.options.equivalence_classes,
#         seed,
#     )
# end
#
#
# function monodromy_solve!(
#     MS::MonodromySolver,
#     seed,
#     progress::Union{Nothing,ProgressMeter.ProgressUnknown};
#     threading::Bool = true,
# )
#     Threads.resize_nthreads!(MS.trackers)
#     t₀ = time_ns()
#     iterations_without_progress = 0 # stopping heuristic
#     # intialize job queue
#     queue = MonodromyJob.(1:nsolutions(MS), 1)
#     thread_queues = [MonodromyJob[] for _ = 1:Threads.nthreads()]
#
#     n = nsolutions(MS)
#     retcode = :none
#     while n < MS.options.target_solutions_count
#         retcode =
#             empty_queue!(queue, thread_queues, MS, t₀, progress; threading = threading)
#
#         if retcode == :done
#             retcode = :success
#             break
#         elseif retcode == :timeout ||
#                retcode == :invalid_startvalue ||
#                retcode == :interrupt
#             break
#         end
#
#         for q in thread_queues
#             append!(queue, q)
#             empty!(q)
#         end
#         isempty(queue) || continue
#
#         # Iterations heuristic
#         n_new = nsolutions(MS)
#         if n == n_new
#             iterations_without_progress += 1
#         else
#             iterations_without_progress = 0
#             n = n_new
#         end
#         if iterations_without_progress == MS.options.max_loops_no_progress &&
#            n_new ≥ MS.options.min_solutions
#             retcode = :heuristic_stop
#             break
#         end
#
#         regenerate_loop_and_schedule_jobs!(queue, MS)
#     end
#     update_progress!(progress, nsolutions(MS), MS.statistics; finish = true)
#
#     retcode
# end
#
# function empty_queue!(
#     queue,
#     thread_queues,
#     MS::MonodromySolver,
#     t₀::UInt,
#     progress;
#     threading::Bool = true,
# )
#     if !threading
#         ntracked = Ref(0)
#         nfailures = Ref(0)
#         n_since_update = 0
#         retcode = :incomplete
#         try
#             while !isempty(queue)
#                 job = pop!(queue)
#                 status = process!(queue, job, MS, progress, ntracked, nfailures, 1)
#                 if status == :done || status == :invalid_startvalue || status == :interrupt
#                     retcode = status
#                     break
#                 end
#                 n_since_update += 1
#
#                 MS.statistics.ntrackedpaths += ntracked[]
#                 MS.statistics.ntrackingfailures += nfailures[]
#                 ntracked[] = nfailures[] = 0
#                 if n_since_update ≥ 4
#                     n_since_update = 0
#                     update_progress!(progress, nsolutions(MS), MS.statistics)
#                 end
#                 # check timeout
#                 if (time_ns() - t₀) > MS.options.timeout * 1e9 # convert s to ns
#                     retcode = :timeout
#                     break
#                 end
#             end
#         catch e
#             if isa(e, InterruptException)
#                 return :interrupt
#             else
#                 rethrow()
#             end
#         end
#         n_since_update > 0 && update_progress!(progress, nsolutions(MS), MS.statistics)
#         retcode
#     else
#         stop = Ref(false)
#         return_status = Ref(:incomplete)
#         start_ntrackedpaths = MS.statistics.ntrackedpaths
#         start_ntrackingfailures = MS.statistics.ntrackingfailures
#         ntrackedpaths = [Ref(0) for _ = 1:Threads.nthreads()]
#         ntrackingfailures = [Ref(0) for _ = 1:Threads.nthreads()]
#         Threads.@threads for job in queue
#             try
#                 !stop[] || continue
#                 tid = Threads.threadid()
#                 status = process!(
#                     thread_queues[tid],
#                     job,
#                     MS,
#                     progress,
#                     ntrackedpaths[tid],
#                     ntrackingfailures[tid],
#                     tid,
#                 )
#                 if status == :done
#                     return_status[] = :done
#                     stop[] = true
#                 elseif status == :invalid_startvalue
#                     return_status[] = :invalid_startvalue
#                     stop[] = true
#                 end
#                 if tid == 1
#                     n_tracked = start_ntrackedpaths + sum(getindex, ntrackedpaths)
#                     n_failures = start_ntrackingfailures + sum(getindex, ntrackingfailures)
#                     MS.statistics.ntrackedpaths = n_tracked
#                     MS.statistics.ntrackingfailures = n_failures
#                     update_progress!(progress, nsolutions(MS), MS.statistics)
#                 end
#                 # check timeout
#                 if (time_ns() - t₀) > MS.options.timeout * 1e9 # convert s to ns
#                     return_status[] = :invalid_startvalue
#                     stop[] = true
#                 end
#                 if stop[]
#                     break
#                 end
#             catch e
#                 if isa(e, InterruptException)
#                     return_status[] = :interrupt
#                     stop[] = true
#                     break
#                 else
#                     rethrow()
#                 end
#             end
#         end
#         empty!(queue)
#         n_tracked = start_ntrackedpaths + sum(getindex, ntrackedpaths)
#         n_failures = start_ntrackingfailures + sum(getindex, ntrackingfailures)
#         MS.statistics.ntrackedpaths = n_tracked
#         MS.statistics.ntrackingfailures = n_failures
#         update_progress!(progress, nsolutions(MS), MS.statistics)
#
#         return_status[]
#     end
# end
#
# function process!(
#     queue,
#     job::MonodromyJob,
#     MS::MonodromySolver,
#     progress,
#     ntrackedpaths::Base.RefValue{Int},
#     ntrackingfailures::Base.RefValue{Int},
#     tid::Integer,
# )
#     x = solutions(MS)[job.id]
#     loop = MS.loops[job.loop_id]
#     retcode = track(MS.trackers[tid], x, loop, ntrackedpaths, ntrackingfailures)
#     is_success(retcode) || return :incomplete
#
#     y = pull_back(MS.problem, current_x(MS.trackers[tid]))
#     add_and_schedule!(MS, queue, y, job) && return :done
#
#     if MS.options.complex_conjugation
#         add_and_schedule!(MS, queue, conj.(y), job) && return :done
#     end
#
#     if !MS.options.equivalence_classes
#         apply_actions(MS.options.group_actions, y) do yᵢ
#             add_and_schedule!(MS, queue, yᵢ, job)
#         end
#     end
#     return :incomplete
# end
#
# """
#     add_and_schedule!(MS, queue, y, job)
#
# Add `y` to the current `node` (if it not already exists) and schedule a new job to the
# `queue`. Returns `true` if we are done. Otherwise `false`.
# """
# function add_and_schedule!(MS::MonodromySolver, queue, y, job::MonodromyJob) where {N,T}
#     lock(MS.solutions_lock)
#     k = add!(MS.solutions, y, Val(true); tol = MS.options.identical_tol)
#     unlock(MS.solutions_lock)
#
#     loop_id = job.loop_id
#     start_sol_id = job.id
#
#
#     if !haskey(MS.statistics.permutations, loop_id)
#         MS.statistics.permutations[loop_id] = Dict{Int,Int}()
#     end
#     if k == NOT_FOUND || k == NOT_FOUND_AND_REAL
#         push!(MS.statistics.permutations[loop_id], start_sol_id => length(MS.solutions))
#     else
#         push!(MS.statistics.permutations[loop_id], start_sol_id => k)
#     end
#
#     if k == NOT_FOUND || k == NOT_FOUND_AND_REAL
#         # Check if we are done
#         isdone(MS.solutions, y, MS.options) && return true
#         push!(queue, MonodromyJob(nsolutions(MS), job.loop_id))
#         # Schedule also on other loops
#         if MS.options.reuse_loops == :random && length(MS.loops) > 1
#             r_loop_id = rand(2:length(MS.loops))
#             if r_loop_id == job.loop_id
#                 r_loop_id -= 1
#             end
#             push!(queue, MonodromyJob(nsolutions(MS), r_loop_id))
#         elseif MS.options.reuse_loops == :all
#             for r_loop_id = 1:length(MS.loops)
#                 r_loop_id != job.loop_id || continue
#                 push!(queue, MonodromyJob(nsolutions(MS), r_loop_id))
#             end
#         end
#     end
#     MS.statistics.nreal += (k == NOT_FOUND_AND_REAL)
#     false
# end
#
# function update_progress!(
#     ::Nothing,
#     nsolutions,
#     statistics::MonodromyStatistics;
#     finish = false,
# )
#     nothing
# end
# function update_progress!(
#     progress,
#     nsolutions,
#     statistics::MonodromyStatistics;
#     finish = false,
# )
#     ProgressMeter.update!(
#         progress,
#         nsolutions,
#         showvalues = (
#             ("# paths tracked", statistics.ntrackedpaths),
#             ("# loops generated", statistics.nparametergenerations),
#             (
#                 "# completed loops without change",
#                 n_completed_loops_without_change(statistics, nsolutions),
#             ),
#             (
#                 "# solutions in current loop",
#                 n_solutions_current_loop(statistics, nsolutions),
#             ),
#             ("# real solutions", statistics.nreal),
#         ),
#     )
#     if finish
#         ProgressMeter.finish!(progress)
#     end
#     nothing
# end
#
# function isdone(solutions::UniquePoints, x, options::MonodromyOptions)
#     options.done_callback(x, solutions.points) ||
#         length(solutions) ≥ options.target_solutions_count
# end
#
# function regenerate_loop_and_schedule_jobs!(queue, MS::MonodromySolver)
#     es = MS.loops[1].edges
#     loop = MonodromyLoop(
#         MS.parameters,
#         length(es),
#         MS.options,
#         weights = !isnothing(es[1].weights),
#     )
#     push!(MS.loops, loop)
#     for id = 1:nsolutions(MS)
#         push!(queue, MonodromyJob(id, length(MS.loops)))
#     end
#     generated_parameters!(MS.statistics, nsolutions(MS))
#     nothing
# end
#
# function update_permutations!(MS::MonodromySolver, job::MonodromyJob, k::Int)
#
#     loop_id = job.loop_id
#     start_sol_id = job.id
#
#
#     # if !haskey(MS.statistics.permutations, loop_id)
#     #     MS.statistics.permutations[loop_id] = Dict{Int,Int}()
#     # end
#     # if k == NOT_FOUND || k == NOT_FOUND_AND_REAL
#     #     push!(MS.statistics.permutations[loop_id], start_sol_id => length(MS.solutions))
#     # else
#     #     push!(MS.statistics.permutations[loop_id], start_sol_id => k)
#     # end
#     nothing
# end
#
#
#
# #############
# ## Results ##
# #############
# """
#     MonodromyResult
#
# The monodromy result contains the result of the `monodromy_solve` computation.
# """
# struct MonodromyResult{T}
#     returncode::Symbol
#     results::Vector{PathResult}
#     parameters::Vector{T}
#     statistics::MonodromyStatistics
#     equivalence_classes::Bool
#     seed::UInt32
# end
#
# Base.iterate(R::MonodromyResult) = iterate(R.results)
# Base.iterate(R::MonodromyResult, state) = iterate(R.results, state)
#
# Base.show(io::IO, ::MIME"application/prs.juno.inline", x::MonodromyResult) = x
# function Base.show(io::IO, result::MonodromyResult{N,T}) where {N,T}
#     println(io, "MonodromyResult")
#     println(io, "==================================")
#     if result.equivalence_classes
#         println(
#             io,
#             "• ",
#             nsolutions(result),
#             " classes of solutions (modulo group action) (",
#             nreal(result),
#             ") real)",
#         )
#     else
#         println(io, "• $(nsolutions(result)) solutions ($(nreal(result)) real)")
#     end
#     println(io, "• return code → $(result.returncode)")
#     println(io, "• $(result.statistics.ntrackedpaths) tracked paths")
#     println(io, "• seed → $(result.seed)")
# end
#
#
# TreeViews.hastreeview(::MonodromyResult) = true
# TreeViews.numberofnodes(::MonodromyResult) = 6
# TreeViews.treelabel(io::IO, x::MonodromyResult, ::MIME"application/prs.juno.inline") =
#     print(
#         io,
#         "<span class=\"syntax--support syntax--type syntax--julia\">MonodromyResult</span>",
#     )
#
# function TreeViews.nodelabel(
#     io::IO,
#     x::MonodromyResult,
#     i::Int,
#     ::MIME"application/prs.juno.inline",
# )
#     if i == 1
#         if x.equivalence_classes
#             print(io, "$(nsolutions(x)) classes of solutions (modulo group action)")
#         else
#             print(io, "$(nsolutions(x)) solutions")
#         end
#     elseif i == 2
#         if x.equivalence_classes
#             print(io, "$(nreal(x)) classes of real solutions")
#         else
#             print(io, "$(nreal(x)) real solutions")
#         end
#     elseif i == 3
#         print(io, "Return code")
#     elseif i == 4
#         print(io, "Statistics")
#     elseif i == 5
#         print(io, "Parameters")
#     elseif i == 6
#         print(io, "Seed")
#     end
# end
# function TreeViews.treenode(r::MonodromyResult, i::Integer)
#     if i == 1
#         return r.solutions
#     elseif i == 2
#         return real_solutions(r)
#     elseif i == 3
#         return r.returncode
#     elseif i == 4
#         return r.statistics
#     elseif i == 5
#         return r.parameters
#     elseif i == 6
#         return r.seed
#     end
#     missing
# end
#
# """
#     is_success(result::MonodromyResult)
#
# Returns true if the monodromy computation achieved its target solution count.
# """
# is_success(result::MonodromyResult) = result.returncode == :success
#
# """
#     is_heuristic_stop(result::MonodromyResult)
#
# Returns true if the monodromy computation stopped due to the heuristic.
# """
# is_heuristic_stop(result::MonodromyResult) = result.returncode == :heuristic_stop
#
# """
#     mapresults(f, result::MonodromyResult; only_real=false, real_tol=1e-6)
#
# Apply the function `f` to all entries of `MonodromyResult` for which the given conditions apply.
#
# ## Example
# ```julia
# # This gives us all solutions considered real (but still as a complex vector).
# real_solutions = mapresults(solution, R, only_real=true)
# ```
# """
# function mapresults(f, R::MonodromyResult; only_real = false, real_tol = 1e-6)
#     [f(r) for r in R.solutions if (!only_real || is_real_vector(r, real_tol))]
# end
#
# """
#     solutions(result::MonodromyResult; only_real=false, real_tol=1e-6)
#
# Return all solutions (as `SVector`s) for which the given conditions apply.
#
# ## Example
# ```julia
# real_solutions = solutions(R, only_real=true)
# ```
# """
# function solutions(R::MonodromyResult; kwargs...)
#     mapresults(identity, R; kwargs...)
# end
#
# """
#     nsolutions(result::MonodromyResult)
#
# Returns the number solutions of the `result`.
# """
# nsolutions(res::MonodromyResult) = length(res.solutions)
#
# """
#     real_solutions(res::MonodromyResult; tol=1e-6)
#
# Returns the solutions of `res` whose imaginary part has norm less than 1e-6.
# """
# function real_solutions(res::MonodromyResult; tol = 1e-6)
#     map(r -> real_vector(r), filter(r -> is_real_vector(r, tol), res.solutions))
# end
#
# """
#     nreal(res::MonodromyResult; tol=1e-6)
#
# Counts how many solutions of `res` have imaginary part norm less than 1e-6.
# """
# function nreal(res::MonodromyResult; tol = 1e-6)
#     count(r -> is_real_vector(r, tol), res.solutions)
# end
#
# """
#     parameters(r::MonodromyResult)
#
# Return the parameters corresponding to the given result `r`.
# """
# parameters(r::MonodromyResult) = r.parameters
#
# """
#     permutations(r::MonodromyResult; reduced=true)
#
# Return the permutations of the solutions that are induced by tracking over the loops. If `reduced=false`, then all permutations are returned. If `reduced=true` then permutations without repetitions are returned.
#
# If a solution was not tracked in the loop, then the corresponding entry is 0.
#
# Example: monodromy loop for a varying line that intersects two circles.
# ```julia
# using LinearAlgebra
# @polyvar x[1:2] a b c
# c1 = (x-[2;0]) ⋅ (x-[2;0]) - 1
# c2 = (x-[-2;0]) ⋅ (x-[-2;0]) - 1
# F = [c1 * c2; a * x[1] + b * x[2] - c]
# S = monodromy_solve(F, [[1.0, 0.0]], [1, 1, 1], parameters = [a;b;c])
#
# permutations(S)
# ```
#
# will return
#
# ```julia
# 2×2 Array{Int64,2}:
#  1  2
#  2  1
# ```
#
# and `permutations(S, reduced = false)` returns
#
# ```julia
# 2×12 Array{Int64,2}:
#  1  2  2  1  1  …  1  2  1  1  1
#  2  1  1  2  2     2  1  2  2  2
# ```
#
# """
# function permutations(r::MonodromyResult; reduced = true)
#
#     π = sort!(collect(r.statistics.permutations), by = first)
#     N = length(solutions(r))
#     if reduced
#         π = unique(map(last, π))
#     else
#         π = map(last, π)
#     end
#     A = zeros(Int, N, length(π))
#     for (j, πᵢ) in enumerate(π)
#         for i = 1:N
#             if haskey(πᵢ, i)
#                 A[i, j] = πᵢ[i]
#             else
#                 A[i, j] = 0
#             end
#         end
#     end
#
#     A
# end
#
#
# ##################
# ## VERIFICATION ##
# ##################
# """
#     verify_solution_completeness(F, res::MonodromyResult;
#                                  parameters=..., trace_tol=1e-6, options...)
#
# Verify that the monodromy computation found all solutions by [`monodromy_solve`](@ref).
# This uses a trace test as described in [^LRS18].
# The trace is a numerical value which is 0 if all solutions are found, for this the
# `trace_tol` keyword argument is used. The function returns `nothing` if some computation
# couldn't be carried out. Otherwise returns a boolean. Note that this function requires the
# computation of solutions to another polynomial system using monodromy. This routine can
# return `false` although all solutions are found if this additional solution set is not
# complete. The `options...` arguments can be everything which is accepted by `solve` and
# `monodromy_solve`.
#
# ### Example
#
# ```
# julia> @polyvar x y a b c;
#
# julia> f = x^2+y^2-1;
#
# julia> l = a*x+b*y+c;
#
# julia> res = monodromy_solve([f,l], [-0.6-0.8im, -1.2+0.4im], [1,2,3]; parameters=[a,b,c])
# MonodromyResult
# ==================================
# • 2 solutions (0 real)
# • return code → heuristic_stop
# • 44 tracked paths
# • seed → 367230
#
# julia> verify_solution_completeness([f,l], res; parameters=[a,b,c], trace_tol = 1e-8)
# [ Info: Compute additional witnesses for completeness...
# [ Info: Found 2 additional witnesses
# [ Info: Compute trace...
# [ Info: Norm of trace: 1.035918995391323e-15
# true
# ```
#
#     verify_solution_completeness(F, S, p; parameters=..., kwargs...)
#
# Verify the solution completeness using the computed solutions `S` to the parameter `p`.
#
#     verify_solution_completeness(TTS, S, W₁₀, p₀::Vector{<:Number}, l₀)
#
# Use the already computeded additional witnesses `W₁₀`. You want to obtain
# `TTS`, `W₁₀` and `l₀` as the output from [`solution_completeness_witnesses`](@ref).
#
#
# [^LRS18]:
#     Leykin, Anton, Jose Israel Rodriguez, and Frank Sottile. "Trace test."
#     Arnold Mathematical Journal 4.1 (2018): 113-125.
# """
# function verify_solution_completeness(F, R::MonodromyResult; kwargs...)
#     verify_solution_completeness(F, solutions(R), R.parameters; kwargs...)
# end
#
# function verify_solution_completeness(
#     F,
#     W₀₁::AbstractVector{<:AbstractVector},
#     p₀::Vector{<:Number};
#     show_progress = true,
#     parameters = nothing,
#     trace_tol = 1e-6,
#     kwargs...,
# )
#     W₁₀, TTS, l₀ = solution_completeness_witnesses(
#         F,
#         W₀₁,
#         p₀;
#         show_progress = show_progress,
#         parameters = parameters,
#         kwargs...,
#     )
#     verify_solution_completeness(TTS, W₀₁, W₁₀, p₀, l₀; show_progress = show_progress)
# end
#
# function verify_solution_completeness(
#     TTS,
#     W₀₁::AbstractVector{<:AbstractVector},
#     W₁₀::AbstractVector{<:AbstractVector},
#     p₀::Vector{<:Number},
#     l₀;
#     show_progress = true,
#     trace_tol = 1e-6,
#     kwargs...,
# )
#     # Combine W₀₁ and W₁₀.
#     S = append!([[x; 0.0] for x in W₀₁], W₁₀)
#     # To verify that we found all solutions we need move in the pencil
#     if show_progress
#         @info("Compute trace...")
#     end
#
#     trace = track_and_compute_trace(TTS, S, l₀; kwargs...)
#     if isnothing(trace)
#         return nothing
#     else
#         show_progress && @info("Norm of trace: $(LinearAlgebra.norm(trace))")
#         LinearAlgebra.norm(trace) < trace_tol
#     end
# end
#
# """
#     solution_completeness_witnesses(F, S, p; parameters=..., kwargs...)
#
# Compute the additional necessary witnesses. Returns a triple `(W₁₀, TTS, l)`
# containing the additional witnesses `W₁₀`, a trace test system `TTS` and
# the parameters `l` for `TTS`.
# """
# function solution_completeness_witnesses(
#     F,
#     W₀₁,
#     p₀::Vector{<:Number};
#     parameters = nothing,
#     show_progress = true,
#     kwargs...,
# )
#     # generate another start pair
#     q₀ = randn(ComplexF64, length(p₀))
#     # Construct the trace test system
#     TTS = TraceTestSystem(SPSystem(F; parameters = parameters), p₀, q₀ - p₀)
#
#     y₁ = solution(solve(F, W₀₁[1]; p₁ = p₀, p₀ = q₀, parameters = parameters, kwargs...)[1])
#     # construct an affine hyperplane l(x) going through y₀
#     l₁ = cis.(2π .* rand(length(y₁)))
#     push!(l₁, -sum(l₁ .* y₁))
#     # This is numerically sometimes not so nice. Let's move to a truly generic one.
#     l₀ = randn(ComplexF64, length(l₁))
#     y₀ = solution(solve(TTS, [y₁; 1]; p₁ = l₁, p₀ = l₀)[1])
#
#     if show_progress
#         @info("Compute additional witnesses for completeness...")
#     end
#
#     R₁₀ = monodromy_solve(TTS, y₀, l₀; max_loops_no_progress = 5, kwargs...)
#     best_result = R₁₀
#     best_params = l₀
#     result_agreed = 0
#     for i = 1:10
#         k₀ = randn(ComplexF64, length(l₀))
#         S_k₀ = solutions(solve(
#             TTS,
#             solutions(R₁₀);
#             start_parameters = l₀,
#             target_parameters = k₀,
#         ))
#         new_result = monodromy_solve(TTS, S_k₀, k₀; max_loops_no_progress = 5)
#         if nsolutions(new_result) == nsolutions(best_result)
#             result_agreed += 1
#         elseif nsolutions(new_result) > nsolutions(best_result)
#             best_result = new_result
#             best_params = k₀
#         end
#         if result_agreed > 2
#             break
#         end
#     end
#
#     W₁₀ = solutions(best_result)
#
#     if show_progress
#         @info("Found $(length(W₁₀)) additional witnesses")
#     end
#     W₁₀, TTS, best_params
# end
#
#
# function track_and_compute_trace(TTS::TraceTestSystem, S, l₀; kwargs...)
#     for i = 1:3
#         TTP = TraceTestPencil(TTS, l₀)
#         R₁ = solve(TTP, S, start_parameters = [0.0], target_parameters = [0.1], kwargs...)
#         R₂ = solve(TTP, S, start_parameters = [0.0], target_parameters = [-.1], kwargs...)
#         if nsolutions(R₁) ≠ length(S) || nsolutions(R₂) ≠ length(S)
#             if i == 3
#                 printstyled("Lost solutions $i times. Abort.\n", color = :red, bold = true)
#                 return nothing
#             end
#             printstyled(
#                 "Lost solutions, need to recompute trace...\n",
#                 color = :yellow,
#                 bold = true,
#             )
#         end
#         s₁ = sum(solutions(R₁))
#         s = sum(S)
#         s₂ = sum(solutions(R₂))
#         # Due to floating point errors, we compute the RELATIVE trace
#         trace = ((s₁ .- s) .- (s .- s₂)) ./ max.(abs.(s₁ .- s), 1.0)
#         return trace
#     end
#     nothing
# end
