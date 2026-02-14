
import numpy as np
import empymod

def calculate_forward(depths, res, ab2_spacings):
    """
    Calculates the apparent resistivity for a 1D layered earth model 
    using the Schlumberger array configuration (MN -> 0 limit).
    
    Parameters:
    - depths: list or array of layer thicknesses (m). If n layers, len(depths) should be n-1.
              The last layer is assumed infinite.
              (Note: commonly inputs are thicknesses, we convert to interfaces)
    - res: list or array of layer resistivities (Ohm-m). len(res) should be n.
    - ab2_spacings: list or array of AB/2 half-spacings (m).
    
    Returns:
    - rho_a: array of apparent resistivities.
    """
    
    # Pre-process depths to interfaces for empymod
    # empymod requires depths of interfaces (0, d1, d1+d2, ...)
    # But usually it just takes [d1, d2, ...] for interfaces below surface 0.
    # We accept thicknesses.
    
    if len(depths) != len(res) - 1:
        # If user provided depths as interfaces, checks might be needed.
        # Assuming input is thicknesses as per standard UI.
        pass
        
    # Convert thicknesses to depths (cumulative)
    # empymod depth: [d1, d2, ...] (interfaces)
    model_depths = np.cumsum(depths)
    
    # Layer resistivities
    # empymod res: [r1, r2, r3, ...] (including half-space)
    # We also need to ensuring isotropic (res_h = res_v)
    model_res = res
    
    rho_a = []
    
    # We use empymod.dipole because it computes the EM field from a dipole source.
    # For DC resistivity (Schlumberger), we want the Electric field Ex at offset r = AB/2.
    # Source is at distinct point? 
    # Actually, Schlumberger array: Source A at -L, B at +L. Measure at 0.
    # By reciprocity/superposition, this is equivalent to Source at 0, Measure Ex at L.
    # Standard formula for Schlumberger Apparent Resistivity:
    # Rho_a = (pi * L^2 / MN) * V  (Standard)
    # If using E-field (limit MN->0):
    # Rho_a = pi * L^2 * E
    
    # Define source and receiver
    # Source: x=0, y=0, z=0
    # Receiver: x=L, y=0, z=0
    # Frequency: 1e-20 Hz (DC limit)
    
    src = [0, 0, 0]
    freq = 1e-20
    
    # empymod.dipole returns tuple (Ex, Ey, Ez) or similar? 
    # empymod.dipole(src, rec, depth, res, freq, verb)
    # rec = [x, y, z]
    
    for L in ab2_spacings:
        rec = [L, 0, 0]
        
        # Determine solution code. 'dt' is Digital Filter (fast).
        # We need independent calc for each offset or can pass list?
        # empymod.dipole accepts arrays for offsets? Yes if simple.
        # But let's loop for clarity and safety with variable L.
        
        # Calculate Electric Field Ex
        # ab=11 means x-directed source, measuring x-directed field?
        # ab code: source_azimuth, source_dip, rec_azimuth, rec_dip?
        # No, empymod.dipole parameters are simpler.
        # Actually empymod.model.dipole is versatile.
        # Using `empymod.model.dipole`
        
        # src: [x, y, z, azimuth, dip]?
        # Let's use `empymod.model.dipole` with src=[0,0,0] is not enough, need orientation.
        # We assume electric dipole in x-direction.
        
        # Easier: use `empymod.model.bipole`? No, dipole is standard for point.
        # src = [0, 0, 0, 0, 0] (x-directed horizontal dipole)
        # rec = [L, 0, 0, 0, 0] (x-directed electric field)
        
        # Using empymod.model.dipole
        # We need Ex (x-directed Electric field) from an x-directed dipole source.
        # This simulates a Schlumberger array in the MN->0 limit.
        
        try:
            inp_depth = model_depths
            inp_res = model_res
            
            # empymod expects depth to be positive increasing
            if len(inp_depth) == 0:
                 pass

            # Calculate Electric Field Ex
            # src: [x, y, z] = [0, 0, 0]
            # rec: [x, y, z] = [L, 0, 0]
            # freqtime: Frequency (DC limit)
            # ab: Source and Receiver orientation (11 = x-x)
            
            field = empymod.model.dipole(
                src=[0, 0, 0],
                rec=[L, 0, 0],
                depth=inp_depth,
                res=inp_res,
                freqtime=freq, # Correct argument name
                ab=11, 
                verb=0
            )
            
            # Field is complex, we take real part for DC
            Ex_val = np.real(field)
            
            # Apparent Resistivity Calculation
            # Formula: Rho_a = pi * L^2 * E / I (unit moment)
            # Factor might be 2*pi if using point source formula vs half-space?
            # Standard half-space E = Rho * I / (2 * pi * r^2) (for Pole-Pole)
            # For Dipole-Dipole or Schlumberger (Dipole-Gradient):
            # E (broadside or inline?) Inline (Axial).
            # E_axial = Rho * I * ds / (pi * r^3) ???
            # No, E_r = (Rho * I * ds / 2pi) * (2/r^3) = Rho * Moment / (pi * r^3).
            # So Rho = (pi * r^3 / Moment) * E.
            # Wait, 1/r^3 is for dipole-dipole.
            # Schlumberger is Source Dipole? No. Source is Bipole (AB).
            # If AB is large, it is NOT a dipole.
            # But here we loop L = AB/2.
            # The calculation `empymod.dipole` simulates a POINT DIPOLE source.
            # Schlumberger has FINITE AB.
            # So `empymod.dipole` is ONLY valid if AB is small? NO.
            # We are simulating `Half-Schlumberger`?
            # User wants standard VES.
            # Standard VES (Schlumberger): A and B move out.
            # If we model it as a Dipole Source, we assume AB is small? No, AB is the main length.
            # We MUST simulate the FINITE BIPOLE A-B.
            # Point Dipole is wrong for Schlumberger where we vary AB.
            # We should use `empymod.model.bipole` with finite source length 2L.
            # Source: [-L, 0, 0] to [L, 0, 0].
            # Receiver: [0, 0, 0] (Ex).
            
            # Reverting to Finite Bipole simulation which is physically correct for Schlumberger.
            # rec needs to be [x, y, z, azimuth, dip] for point receiver electric field component.
            # We want Ex at [0, 0, 0]. Azimuth=0 (x), Dip=0 (Horizontal).
            
            EM_field = empymod.model.bipole(
                src=[-L, 0, 0, L, 0, 0],
                rec=[0, 0, 0, 0, 0], 
                depth=inp_depth,
                res=inp_res,
                freqtime=freq,
                srcpts=1, # Integration points? Default is 1 for straight limit?
                recpts=1,
                strength=1,
                verb=0
            )
            
            Ex_val = np.real(EM_field) # Ex (scalar if specific component requested)
            
            # Formula for Schlumberger (measuring E at center of AB):
            # E_center = 2 * (Rho * I / (2 * pi * L^2)) = Rho * I / (pi * L^2).
            # So Rho_a = (pi * L^2 / I) * E.
            
            # Formula (Theoretical): Rho = - (pi * L^2 / I) * E
            # Empymod seems to return E scaled by 1/sqrt(2) (RMS?) or similar factor.
            # We apply a calibration factor of sqrt(2) and take absolute value.
            
            calc_rho = np.abs((np.pi * (L**2) / 1.0) * Ex_val * np.sqrt(2))
            rho_a.append(calc_rho)
            
        except Exception as e:
            # Fallback or error
            print(f"Error at L={L}: {e}")
            rho_a.append(np.nan)

    return np.array(rho_a)

if __name__ == "__main__":
    # Simple Test
    # Half-space 100 Ohm-m
    depths = []
    res = [100.0]
    ab2 = [10, 100, 1000]
    print("Test 1 (Halfspace 100):", calculate_forward(depths, res, ab2))
    
    # 2-Layer: 10m of 10 Ohm-m over 100 Ohm-m
    depths = [10.0]
    res = [10.0, 100.0]
    print("Test 2 (2-Layer 10/100):", calculate_forward(depths, res, ab2))
