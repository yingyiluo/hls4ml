#
# Export few environment variables to compile the example with Mentor's
# provided compiler and libraries.
#

# Base directory for CAD tools.
setenv CAD_PATH /vol/mentor/Catapult_Synthesis_10.4a

# We do need Catapult HLS (high-level synthesis).
setenv PATH ${PATH}:${CAD_PATH}/Mgc_home/bin:/vol/mentor/questa_sim-2019.3_2/questasim/bin

# Let's use the SystemC headers and library provided with Catapult HLS.
setenv SYSTEMC ${CAD_PATH}/Mgc_home/shared

# We do need Synopsys Design Compiler (logic synthesis).
# export PATH="${CAD_PATH}/syn/bin":$PATH

# Makefiles and scripts from Catapult HLS may rely on this variable.
setenv MGC_HOME ${CAD_PATH}/Mgc_home

####
# We do need licensing for this example.
####

# Mentor Graphics
setenv MGLS_LICENSE_FILE 1717@mentorlm.eecs.northwestern.edu
