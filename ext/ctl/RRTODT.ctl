
// <ACEStransformID>RRT.a1.0.3</ACEStransformID>
// <ACESuserName>ACES 1.0 - RRT</ACESuserName>

// 
// Reference Rendering Transform (RRT)
//
//   Input is ACES
//   Output is OCES
//



import "ACESlib.Utilities";
import "ACESlib.Transform_Common";
import "ACESlib.RRT_Common";
import "ACESlib.ODT_Common";
import "ACESlib.Tonescales";


/* --- ODT Parameters --- */
// const Chromaticities DISPLAY_PRI = REC709_PRI;
// const float XYZ_2_DISPLAY_PRI_MAT[4][4] = XYZtoRGB(DISPLAY_PRI,1.0);

// const float DISPGAMMA = 2.4; 
// const float L_W = 1.0;
// const float L_B = 0.0;


void main 
( 
  input varying float rIn,
  input varying float gIn,
  input varying float bIn,
  input varying float aIn,
  output varying float rOut,
  output varying float gOut,
  output varying float bOut,
  output varying float aOut
)
{
    // --------------------------------------------------------------
    // Post-RRT. Input is in linear-encoded AP1 color space.
    // --------------------------------------------------------------
    float rgbPre[3] =  {rIn, gIn, bIn};

    // --- Apply the tonescale independently in rendering-space RGB --- //
    float rgbPost[3];
    rgbPost[0] = segmented_spline_c5_fwd( rgbPre[0]);
    rgbPost[1] = segmented_spline_c5_fwd( rgbPre[1]);
    rgbPost[2] = segmented_spline_c5_fwd( rgbPre[2]);

    // --------------------------------------------------------------
    // Pre-ODT.
    // --------------------------------------------------------------

    // Apply the tonescale independently in rendering-space RGB
    rgbPost[0] = segmented_spline_c9_fwd( rgbPost[0]);
    rgbPost[1] = segmented_spline_c9_fwd( rgbPost[1]);
    rgbPost[2] = segmented_spline_c9_fwd( rgbPost[2]);

    // Scale luminance to linear code value
    float linearCV[3];
    linearCV[0] = Y_2_linCV( rgbPost[0], CINEMA_WHITE, CINEMA_BLACK);
    linearCV[1] = Y_2_linCV( rgbPost[1], CINEMA_WHITE, CINEMA_BLACK);
    linearCV[2] = Y_2_linCV( rgbPost[2], CINEMA_WHITE, CINEMA_BLACK);

    // Assign OCES RGB to output variables (OCES)
    rOut = linearCV[0];
    gOut = linearCV[1];
    bOut = linearCV[2];
    aOut = aIn;
}