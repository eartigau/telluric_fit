#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Wrap file for LBL recipes

Created on 2024-07-09 16:04:51.776

@author: Neil Cook, Etienne Artigau, Charles Cadieux, Thomas Vandal, Ryan Cloutier, Pierre Larue

user: eartigau@amakihi.local
lbl version: 0.63.010
lbl date: 2024-07-09
"""
import os

from lbl import lbl_wrap



# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # set up parameters
    rparams = dict()
    # -------------------------------------------------------------------------
    # LBL parameters
    # -------------------------------------------------------------------------
    # You may also add any constant here to override the default value
    #     (see README for details) - this is NOT recommended for non developers
    #   Note this may have undesired affects as these parameters apply globally
    #     for all LBL recipes
    # -------------------------------------------------------------------------
    # This is the instrument name
    #   Currently supported instruments are 
	#		SPIROU
	#		HARPS
	#		ESPRESSO
	#		CARMENES
	#		NIRPS_HA
	#		NIRPS_HE
	#		HARPSN
	#		MAROONX
	#		SOPHIE
    rparams['INSTRUMENT'] = 'NIRPS_HE'
    #   Data source must be as follows: 
	#		SPIROU: APERO or CADC
	#		NIRPS_HA: APERO or CADC or ESO
	#		NIRPS_HE: APERO or CADC or ESO
	#		HARPS: ORIG or ESO
	#		CARMENES: None
	#		ESPRESSO: None
	#		HARPSN: ORIG or ESO
	#		MAROONX: RED or BLUE
	#		SOPHIE: None
    rparams['DATA_SOURCE'] = 'CADC'
    # The data directory where all data is stored under - this should be an
    #    absolute path
    rparams['DATA_DIR'] = '/home/eartigau/projects/rrg-rdoyon/eartigau/tapas/test_fit/lbl_NIRPS/'
    # The input file string (including wildcards) - if not set will use all
    #   files in the science directory (for this object name)
    rparams['INPUT_FILE'] = '*t.fits'
    # The input science data are blaze corrected
    rparams['BLAZE_CORRECTED'] = False
    # Override the blaze filename
    #      (if not set will use the default for instrument)
    # rparams['BLAZE_FILE'] = 'blaze.fits'
    # -------------------------------------------------------------------------
    # science criteria
    # -------------------------------------------------------------------------
    # The object name (this is the directory name under the /science/
    #    sub-directory and thus does not have to be the name in the header
    rparams['OBJECT_SCIENCE'] = ['TOI4552_skypca_v5_model']
    # This is the template that will be used or created (depending on what is
    #   run)
    rparams['OBJECT_TEMPLATE'] = ['TOI4552_skypca_v5_model']
    # The data type (either SCIENCE or FP or LFC)
    rparams['DATA_TYPES'] = ['SCIENCE']*len(rparams['OBJECT_SCIENCE'])
    # This is the object temperature in K - used for getting a stellar model
    #   for the masks it only has to be good to a few 100 K
    rparams['OBJECT_TEFF'] = [3000]*len(rparams['OBJECT_SCIENCE'])
    # -------------------------------------------------------------------------
    # what to run and skip if already on disk
    # -------------------------------------------------------------------------
    # Whether to run the telluric cleaning process (NOT recommended for data
    #   that has better telluric cleaning i.e. SPIROU using APERO)
    rparams['RUN_LBL_TELLUCLEAN'] = False
    # Whether to create templates from the data in the science directory
    #   If a template has been supplied from elsewhere this set is NOT required
    rparams['RUN_LBL_TEMPLATE'] = True
    # Whether to create a mask using the template created or supplied
    rparams['RUN_LBL_MASK'] = True
    # Whether to run the LBL compute step - which computes the line by line
    #   for each observation
    rparams['RUN_LBL_COMPUTE'] = True
    # Whether to run the LBL compile step - which compiles the rdb file and
    #   deals with outlier rejection
    rparams['RUN_LBL_COMPILE'] = True
    # whether to skip observations if a file is already on disk (useful when
    #   adding a few new files) there is one for each RUN_XXX step
    #   - Note cannot skip tellu clean
    rparams['SKIP_LBL_TEMPLATE'] = True
    rparams['SKIP_LBL_MASK'] = True
    rparams['SKIP_LBL_COMPUTE'] = True
    rparams['SKIP_LBL_COMPILE'] = False
    # -------------------------------------------------------------------------
    # LBL settings
    # -------------------------------------------------------------------------
    # You can change any setting in parameters (or override those changed
    #   by specific instruments) here
    # -------------------------------------------------------------------------
    # Advanced settings
    #   Do not use without contacting the LBL developers
    # -------------------------------------------------------------------------
    # Dictionary of table name for the file used in the projection against the
    #     derivative. Key is to output column name that will propagate into the
    #     final RDB table and the value is the filename of the table. The table
    #     must follow a number of characteristics explained on the LBL website.
    rparams['RESPROJ_TABLES'] = dict()
    rparams['RESPROJ_TABLES']['DTEMP3000'] = 'temperature_gradient_3000.fits'

    # Rotational velocity parameters, should be a list of two values, one being
    #     the epsilon and the other one being the vsini in km/s as defined in the
    #     PyAstronomy.pyasl.rotBroad function
    # rparams['ROTBROAD'] = []

    # turn on plots
    rparams['PLOT'] = False

    # -------------------------------------------------------------------------
    # Run the wrapper code using the above settings
    # -------------------------------------------------------------------------
    # run main
    lbl_wrap(rparams)

# =============================================================================
# End of code
# =============================================================================
