
#* check units on kB and hBar from unitful
function bose_einstein(freq::Real, temp::Real, kB::Real)
    return 1 / (exp((ħ * freq) / (kB * temp)) - 1)
end

function quantum_amplitudes(freqs, masses, temp)
    numerator = ħ .* (2*)
end

function classical_amplitudes(freqs, masses, kB, temp)
    return @. (1/freqs)*sqrt((kB*temp)/masses)
end

function self_consistent_IFC_loop(sys_eq::SuperCellSystem, temp::Real, 
                                    pot::Potential, n_configs::Int, 
                                    outpath::String, mode::Symbol)


    # Calculate 0K IFCs to initialize loop
    

end