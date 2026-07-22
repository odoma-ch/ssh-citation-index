#!/usr/bin/env python3
"""Test script for concurrent queue operations with mixed task types.

Tests:
1. Multiple text extraction jobs (PDF -> text) [default queue]
2. Multiple reference extraction jobs (text -> references) [llm-tasks queue]
3. Multiple reference parsing jobs (references -> structured) [llm-tasks queue]
4. Mixed workload with all three types
"""

import json
import time
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List
import io

API_BASE = "https://citation-index-api-graphia-app1-staging.apps.bst2.paas.psnc.pl"

# Timeout settings (matching .env configuration + buffer for polling overhead)
TIMEOUT_TEXT_EXTRACTION = 300 + 60  # 5min task + 1min buffer = 6min polling timeout
TIMEOUT_REFERENCE_EXTRACTION = 900 + 120  # 15min task + 2min buffer = 17min polling timeout
TIMEOUT_REFERENCE_PARSING = 900 + 120  # 15min task + 2min buffer = 17min polling timeout

# Sample texts for reference extraction from EXCITE benchmark markdown files
SAMPLE_TEXTS = [
    # From 1181_marker.md - German sociology paper on interpretative flexibility
    """## Literatur

Aron, Raymond/Dominique Schnapper (1988): Power, modernity, and sociology : selected sociological writings. Aldershot, Hants, England

Brookfield, Vt., USA: E. Elgar ;

- Collins, Harry (2004): Gravity's shadow : the search for gravitational waves. Chicago: University of Chicago Press.
- Collins, Harry M. (1981): Stages in the Empirical Programme of Relativism. In: Social Studies of Science, 11 S. 3-10.
- Collins, Harry M. (1985): Changing order : replication and induction in scientific practice, London u.a.: Sage.

Dosi, Giovanni (1982): Technological Paradigms and Technological Trajectories. In: Research Policy, 11 S. 147-162.

Kuhn, Thomas S. (1976): Die Struktur wissenschaftlicher Revolutionen. Frankfurt/Main: Suhrkamp.

Luhmann, Niklas (1990): Die Wissenschaft der Gesellschaft, Frankfurt/Main: Suhrkamp.""",
    
    # From 22622_marker.md - Book review on Dolly the sheep cloning
    """#### **Conclusion**

Effective interventions should be predicated on

precepts and values that are appropriate, relevant and consistent with the cultural realities of these youth. Employing a Eurocentric paradigm as the foundation of conventional approaches to African American youth who are substance abusers is tantamount to having the wrong calabash—that is, if our intention is to support the optimal functioning of the adolescent within an interconnected spiritual, cultural, social, and psychological Universe.

As noted by Nobles and Goddard (1993), a maximally effective intervention must be "grounded in the same culture that gives meaning to human needs and functions; ...responds to the conditions in the target community; and addresses the real (actual) problems experienced and defined by the community" (p. 11). Traditional African philosophy and values provide the foundation for a transformative approach to our work, one that not only affirms the worth of those adolescents with whom we work, but also affirms our worth in the collaborative journey toward remembering our interdependence as we assist them (and in turn, ourselves) in re-membering, restoring harmony and balance in those out-of-order aspects of their (and in turn, our own) interconnected lives.

{9}------------------------------------------------

# *References*

Ani, M. (1994). Yurugu: an African-centered critique of European culture, thought and behavior. Trenton, NJ: Africa World Press. Azibo, D. A. (2014). The Azibo Nosology II: Epexegesis and 25th anniversary update: 55 culture-focused mental disorders suffered by African descent people. *Journal o f Pan African Studies,* 7(5), 32-176. Retrieved January 22, 2019, from

[www.jpanafrican.org/docs/vol7no4/5-nov-azibo-noso.pdf](http://www.jpanafrican.org/docs/vol7no4/5-nov-azibo-noso.pdf).

- Asante, M.K. (1987). The Afrocentric idea. Philadephia: Temple University Press.
- Bent-Goodley, T.B. (2005). An African-centered approach to domestic violence. *Families in Society, 86(2),* 197-206.

Betz, C., Mihalic, D., Pinto, M.E. & Raffa, R.B. (2000). Could a common biochemical mechanism underlie addictions? *Journal o f Clinical Pharmacy and Therapeutics,* 25, 11-20.

Boyd-Franklin,N. (2010). Incorporating spirituality and religion into the treatment of African Americana clients. Counseling Psychologist. 38:7, pp. 1-25. Downloadedfromtcp.sagepub.com.

Brandon, T.H., Herzog, T.A., Irvin, J.E. & Gwaltney, C.J. (2004). Cognitive and social learning models of drug dependence: implications for the assessment of tobacco dependence in adolescents, *Societyfor the Study o f Addiction,* 99(Supplement 1), 51-77.

Brookins, C.C. (1999). Afrikan/Community psychology: Exploring the foundations of a progressive paradigm, in R.L. Jones (Ed.), *Advances in African American psychology,* Hampton, VA: Cobb & Henry Publishers, 27-50.

Clarke, J.H. (1990). African American historians and the reclaiming of African history, in M.K. Asante & K.W. Asante (Ed.), *African culture.* Trenton, NJ: Africa World Press.

Cloninger, C. R. (1987). Neurogenetic adaptive mechanisms in alcoholism. *Science, 236,* 410-436.

Dean, R.G. (2001). The myth of cross-cultural competence. *Families in Society, 82(6),* 623-630.

Diallo, Y. (1998). *Dombaa folee* (CD). New York: Relaxation Company.

Dzobo, N.K. (1997). *African proverbs.* Accra, Ghana: Life Press, Ltd.

Harvey A.R. & Rauch, J.B. (1997). Africentric youth and family rites of passage program: Promoting resilience among at-risk African American youths. *Social Work, 49,* 65-74.

Jackson, M. S. (1995). Afrocentric treatment of African American women and their children in residential chemical dependency program. Journal ofBlack Studies, 26(1), 17-30.

Jones, J.M. (1991). Racism, a cultural analysis of the problem. In R.L. Jones (Ed.), *Black psychology, third edition.* Berkeley, CA: Cobb & Henry.

Kumpfer, K.L. & Alvarado, R. (2003). Family-strengthening approaches for the prevention of youth problem behaviors. *American Psychologist,* 55(6/7), 457-466.

Lilja, J., Larsson, S., Wilhelmsen, B.U. & Hamilton, D. (2003). Perspectives on preventing adolescent substance use of misuse, *Substance Use & Misuse,* 35(10), 1491-1530.

Longshore, D., Grills, C., Annon, K. & Grady, R., (1998). Promoting recovery from drug abuse, an Africentric intervention. *Journal o f Black Studies,* 25(3), 319-333.

Longshore, D., Grills, C., Anglin, M.D. & Annon, K. (1997). Desire for help among African American drug users. *Journal o f Drug Issues, 27(A),* 755-771.

Martin, E. & Martin, J. (2002). *Spirituality and the black helping tradition.* Washington D.C.: N A S W Press.

Mazama, M.A. (2002). Afrocentricity and African spirituality, *Journal o f Black Studies, 33(2),* 218-234.

Mbiti, J.S. (1975). *Introduction to African religions.* New York: Praeger.

Mensah, A.J. (1993). *Rites o f passage and initiation processes with Akan culture.* Unpublished manuscript.

Mfuni, T. (2005). *Counseling people to consciousness.* New York Amsterdam News. 96(50).

Moduka, M.l. (1999). *Affirming unity in diversity in education: healing with ubuntu.* Capetown: Juta & Co.

Moore, S. E., Madison-Comore, O. & Moore, J.L. (2003). An Afrocentric approach to substance abuse treatment with adolescent African American males: two case examples. *The Western Journal o f Black Studies,* 27(4), 219-230.

Myers, L. (2013). Substance Use Among Rural African American Adolescents: Identifying Risk and Protective Factors. Child & Adolescent Social Work Journal, 30(1), 79-93. doi: 10.1007/sl 0560-012-0280-2

Myers, L.G. & Speight, S. (2010). Refraiming mental health psychological well-being among persons of African descent: African/Black psychology meeting the challenges of fractured social and cultural realities. Journal of Pan African Studies, 3:8, pp. 66-82.

Myers, L. J. (1988). *An Afrocentric worldview: Introduction to an optimal psychology.* Dubuque, Iowa: Kendall/Hunt Publishers.

Nobles, W.W. & Goddard, L.L. (1993). *An African-centered model o f prevention fo r African American youth at high risk.* Retrieved October 11, 2018 from [http://www.iasbflc.org/Articles/AfricanModel/africanmodel01.htm.](http://www.iasbflc.org/Articles/AfricanModel/africanmodel01.htm)

Nobles, W.W., Goddard, L.L., Cavil, W.E. & George, P.Y. (1987). *African American families.* Oakland, CA: Institute for the Advancement of the Black Family.""",
    
    # From 4605_marker.md - German paper on North Korean workers in Russia
    """The foliar application of Fe inhibited the ferric reductase activity in Col-0 (**[Figure 4](#page-5-0)A**) and in the *pho2* mutant (**[Figure 4](#page-5-0)C**) while it had less effect in the *opt3-2* mutant (**[Figure 4B](#page-5-0)**). This latter effect agrees with previous results (García et al., 2013). Curiously, the foliar application of P also inhibited the ferric reductase activity in Col-0 plants, less than the Fe application (**[Figure 4](#page-5-0)A**), but had no effect on *pho2* (**[Figure 4](#page-5-0)C**) and hardly in *opt3-2* (**[Figure 4B](#page-5-0)**). These results suggest that there are Fe and P shoot-derived signals

{10}------------------------------------------------

moving through the phloem that are inhibitory and that both Fe and P signals are interrelated. This interrelation is further supported when analysing phosphatase activity and the expression of one of its encoding genes, *AtPAP17*. As shown in **[Figure 5](#page-6-0)**, the foliar application of either Fe or P drastically inhibited the phosphatase activity and *AtPAP17* expression in Col-0. However, neither Fe nor P application appreciably inhibited these P responses in either mutant (**Figures 6** and **7**).

Despite the existence of common signals involved in the activation of responses to both nutrient deficiencies, like ethylene (**[Figure 3](#page-4-2)**), and the cross talk between shoot-derived signals (see previous paragraph), some results presented in this work suggest the existence of specific signals that block the activation of the responses to one deficiency when the deficiency is caused by the other element. For example, the expression of Fe acquisition genes is almost constitutively activated in the *opt3-2* mutant (**Figures 8A**,

increase under Fe or P sufficiency. For more details, see text. (à: promotion; 〒: inhibition). Working model based on [Chiou and Lin \(2011\)](#page-11-8) and García et al. (2018).

{11}------------------------------------------------

**B**) while the expression of P acquisition genes is not (**Figures 8C**, **D**). This suggests that the absence of the phloem-Fe signal related to OPT3 depresses the expression of Fe acquisition genes (García et al., 2013) but not that of P acquisition genes, probably because there are specific P signals that block it.

According to the results obtained in this study we propose a Working Model to explain the role of ethylene and P-related and Fe-related phloem signals on the regulation of Fe and P acquisition genes (**Figure 9**).Once inside roots, Fe (black arrows) is translocated to leaves through the xylem, bound to citrate (provided by the FRD3 transporter). In shoots, some Fe can enter the phloem through the OPT3 transporter, and moves back to roots probably bound to a chelating agent (Fe)?. In roots, this Fe? can be sensed by the Brutus protein (BTS) that blocks the expression of the Fe acquisition genes *FRO2* and *IRT1*, probably because it inhibits the synthesis of ethylene (ET), which has been involved in their upregulation. P (blue arrows) is absorbed through P transporters, like Pht1;4, and then loaded into the xylem through transporters like PHO1. Under P deficiency, miR399 in shoots increases and moves through the phloem to roots where it suppresses PHO2. PHO2 participates in the degradation of Pht1;4 and PHO1, and probably inhibits ethylene synthesis. Consequently, under P deficiency, the suppression of PHO2 by miR399 can permit the stabilization of Pht1;4 and PHO1, and the synthesis of ethylene, which has been involved in the upregulation of *Pht1;4* and *PAP17*, encoding a phosphatase (PAP).

In conclusion, the results obtained in this work further support the cross talk between Fe and P deficiency responses since Fe deficiency induces the expression of P acquisition genes and the phosphatase activity, and P deficiency induces the expression of Fe acquisition genes and the ferric reductase activity. In most cases, like the induction of Fe acquisition genes and ferric reductase activity by P deficiency, this occurs very transitorily, probably due to the existence of Fe-related inhibitory signals

# REFERENCES

- <span id="page-11-5"></span>Abel, S. (2011). Phosphate sensing in root development. *Curr. Opin. Plant Biol.* 14, 303–309. doi: [10.1016/j.pbi.2011.04.007](https://doi.org/10.1016/j.pbi.2011.04.007)
- <span id="page-11-4"></span>Bari, R., Pant, B. D., Stitt, M., and Scheible, W. R. (2006). PHO2, microRNA399, and PHR1 define a phosphate- pathway in plants. *Plant Physiol.* 141, 988–999. doi: [10.1104/pp.106.079707](https://doi.org/10.1104/pp.106.079707)
- <span id="page-11-7"></span>Borch, K., Bouma, T. J., Lynch, J. P., and Brown, K. M. (1999). Ethylene: a regulator of root architectural responses to soil phosphorus availability. *Plant Cell Environ.* 22, 425–431. doi: [10.1046/j.1365-3040.1999.00405.x](https://doi.org/10.1046/j.1365-3040.1999.00405.x)
- <span id="page-11-0"></span>Bhosale, R., Giri, J., Pandey, B. K., Giehl, R. F., Hartmann, A., Traini, R., et al. (2018). A mechanistic framework for auxin dependent *Arabidopsis* root hair elongation to low external phosphate. *Nat. Commun.* 9 (1), 1409–1417. doi: [10.1038/s41467-018-03851-3](https://doi.org/10.1038/s41467-018-03851-3)
- <span id="page-11-1"></span>Brumbarova, T., Bauer, P., and Ivanov, R. (2015). Molecular mechanisms governing *Arabidopsis* iron uptake. *Trends Plant Sci.* 20, 124–133. doi: [10.1016/j.](https://doi.org/10.1016/j.tplants.2014.11.004) [tplants.2014.11.004](https://doi.org/10.1016/j.tplants.2014.11.004)
- <span id="page-11-12"></span>Buhtz, A., Pieritz, J., Springer, F., and Kehr, J. (2010). Phloem small RNAs, nutrient stress responses, and systemic mobility. *BMC Plant Biol.* 10, 64. doi: [10.1186/1471-2229-10-64](https://doi.org/10.1186/1471-2229-10-64)
- <span id="page-11-10"></span>Connolly, E. L., Campbell, N. H., Grotz, N., Prichard, C. L., and Guerinot, M. L. (2003). Overexpression of the FRO2 ferric chelate reductase confers tolerance to growth on low iron and uncovers posttranscriptional control. *Plant Physiol.* 133, 1102–1110. doi: [10.1104/pp.103.025122](https://doi.org/10.1104/pp.103.025122)
- <span id="page-11-11"></span>Chen, W. W., Yang, J. L., Qin, C., Jin, C. W., Mo, J. H., Ye, T., et al. (2010). Nitric oxide acts downstream of auxin to trigger root ferric-chelate reductase activity

and because of the post-transcriptional regulation of FRO2 by Fe. The cross talk between both deficiencies could be related to the existence of common signals, like ethylene, implicated in the activation of their responses. Besides ethylene, the results obtained with the foliar application of Fe or P show additional interactions between P and Fe inhibitory signals coming from the shoots, and suggest that Fe- or P-related phloem signals could interact with ethylene in the regulation of the responses to their deficiencies.

# AUTHOR CONTRIBUTIONS

CL and FR designed the experiments after discussions with JG-M and MG; RP and MG conducted the laboratory work; ÁZ and EB determined ACC; AS, EA, RP-V, and CL wrote the manuscript.

# FUNDING

This work was supported by the European Regional Development Fund from the European Union, the "Ministerio de Economía y Competitividad" (Projects AGL2013- 40822-R and AGL2015- 65104-P); the "Junta de Andalucía" (Research Groups AGR115 and BIO159) and the economic contribution of the Company TimacAgro Spain (Roullier Group).

# ACKNOWLEDGMENTS

We thank Dr. Joaquin Romero, of the University of Córdoba (Spain), for his support and valuable contribution in the statistical analysis of results.

in response to iron deficiency in *Arabidopsis thaliana. Plant Physiol.* 154, 810– 819. doi: [10.1104/pp.110.161109](https://doi.org/10.1104/pp.110.161109)

- <span id="page-11-8"></span>Chiou, T. J., and Lin, S. I. (2011). Network in sensing phosphate availability in plants. *Annu Rev Plant Biol.* 62, 185–206. doi: [10.1146/](https://doi.org/10.1146/annurev-arplant-042110-103849) [annurev-arplant-042110-103849](https://doi.org/10.1146/annurev-arplant-042110-103849)
- <span id="page-11-6"></span>de Kock, P. C. (1955). Iron nutrition of plants at high pH. *Soil Sci.* 79, 167–175. doi: [10.1097/00010694-195503000-00001](https://doi.org/10.1097/00010694-195503000-00001)
- <span id="page-11-9"></span>Delhaize, E., and Randall, P. J. (1995). Characterization of a phosphate accumulator mutant of *Arabidopsis thaliana. Plant Physiol.* 107, 207–213. doi: [10.1104/](https://doi.org/10.1104/pp.107.1.207) [pp.107.1.207](https://doi.org/10.1104/pp.107.1.207)
- <span id="page-11-2"></span>del Pozo, J. C., Allona, I., Rubio, V., Leyva, A., de la Peña, A., Aragoncillo, C., et al. (1999). A type 5 acid phosphatase gene from *Arabidopsis thaliana* is induced by phosphate starvation and by som other types of phosphate mobilising/oxidative stress conditions. *Plant J.* 19, 579–589. doi: [10.1046/j.1365-313X.1999.00562.x](https://doi.org/10.1046/j.1365-313X.1999.00562.x)
- <span id="page-11-3"></span>Franco-Zorrilla, J. M., González, E., Bustos, R., Linhares, F., Leyva, A., and Paz-Ares, J. (2004). The transcriptional control of plant responses to phosphate limitation. *J. Exp. Bot.* 55, 285–293. doi: [10.1093/jxb/erh009](https://doi.org/10.1093/jxb/erh009)
- García, M. J., Lucena, C., Romera, F. J., Alcántara, E., and Pérez-Vicente, R. (2010). Ethylene and nitric oxide involvement in the up-regulation of key genes related to iron acquisition and homeostasis in *Arabidopsis. J. Exp. Bot.* 61, 3885–3899. doi: [10.1093/jxb/erq203](https://doi.org/10.1093/jxb/erq203)
- García, M. J., Suárez, V., Romera, F. J., Alcántara, E., and Pérez-Vicente, R. (2011). A new model involving ethylene, nitric oxide and Fe to explain the regulation of Fe-acquisition genes in Strategy I plants. *Plant Physiol. Biochem*. 49, 537– 544. doi: [10.1016/j.plaphy.2011.01.019](https://doi.org/10.1016/j.plaphy.2011.01.019)

{12}------------------------------------------------

- García, M. J., Romera, F. J., Stacey, M. G., Stacey, G., Villar, E., Alcántara, E., et al. (2013). Shoot to root communication is necessary to control the expression of iron-acquisition genes in Strategy I plants. *Planta* 237, 65–75. doi: [10.1007/](https://doi.org/10.1007/s00425-012-1757-0) [s00425-012-1757-0](https://doi.org/10.1007/s00425-012-1757-0)
- García, M.J., García-Mateo, M.J., Lucena, C., Romera, F.J.,Rojas, C.L., Alcántara, E., et al. (2014). Hypoxia and bicarbonate could block the expression of iron acquisition genes in Strategy I plants by affecting ethylene synthesis and signaling in different ways. *Physiol. Plant*. 150, 95–106. doi: [10.1111/ppl. 12076](https://doi.org/10.1111/ppl. 12076)
- García, M. J., Romera, F. J., Lucena, C., Alcántara, E., and Pérez-Vicente, R. (2015). Ethylene and the regulation of physiological and morphological responses to nutrient deficiencies. *Plant Physiol.* 169, 51–60. doi: [10.1104/pp.15.00708](https://doi.org/10.1104/pp.15.00708)
- García, M. J., Corpas, F. J., Lucena, C., Alcántara, E., Pérez-Vicente, R., Zamarreño, Á.M., et al. (2018). A shoot Fe pathway requiring the OPT3 transporter controls GSNO reductase and ethylene in *Arabidopsis thaliana* roots. *Front. Plant Sci.* 9, 1325–1341. doi: [10.3389/fpls.2018.01325](https://doi.org/10.3389/fpls.2018.01325)
- <span id="page-12-29"></span>Graziano, M., and Lamattina, L. (2007). Nitric oxide accumulation is required for molecular and physiological responses to iron deficiency in tomato roots. *Plant J.* 52, 949–960. doi: [10.1111/j.1365-313X.2007.03283.x](https://doi.org/10.1111/j.1365-313X.2007.03283.x)
- <span id="page-12-21"></span>Ham, B. K., Chen, J., Yan, Y., and Lucas, W. J. (2018). Insights into plant phosphate sensing and signaling. *Curr. Opi. Biotech* 49, 1–9. doi: [10.1016/j.copbio.2017.07.005](https://doi.org/10.1016/j.copbio.2017.07.005)
- <span id="page-12-15"></span>Henry, J. B., McCall, I., Jackson, B., and Whipker, B. E. (2017). Growth response of herbaceous ornamental to phosphorus fertilization. *Hortscience* 52, 1362– 1367. doi: [10.21273/HORTSCI12256-17](https://doi.org/10.21273/HORTSCI12256-17)
- <span id="page-12-13"></span>Hirsch, J., Marin, E., Floriani, M., Chiarenza, S., Richaud, P., Nussaume, L., et al. (2006). Phosphate deficiency promotes modification of iron distribution in *Arabidopsis* plants. *Biochimic.* 88, 1767–1771. doi: [10.1016/j.biochi.2006.05.007](https://doi.org/10.1016/j.biochi.2006.05.007)
- <span id="page-12-7"></span>Ivanov, R., Brumbarova, T., and Bauer, P. (2012). Fitting into the harsh reality: regulation of iron-deficiency responses in dicotyledonous plants. *Mol. Plant* 5, 27–42. doi: [10.1093/mp/ssr065](https://doi.org/10.1093/mp/ssr065)
- <span id="page-12-23"></span>Kumar, S., Verma, S., and Trivedi, P. K. (2017). Involvement of Small RNAs in phosphorus and sulfur sensing, and stress: current update. *Front. Plant. Sci.* 8, 285. doi: [10.3389/fpls.2017.00285](https://doi.org/10.3389/fpls.2017.00285)
- <span id="page-12-17"></span>Lei, M., Zhu, C., Liu, Y., Karthikeyan, A. S., Bressan, R. A., Raghothama, K. G., et al. (2011). Ethylene is involved in regulation of phosphate starvation-induced gene expression and production of acid phosphatases and anthocyanin in *Arabidopsis*. *New Phytol.* 189, 1084–1095. doi: [10.1111/j.1469-8137.2010.03555.x](https://doi.org/10.1111/j.1469-8137.2010.03555.x)
- <span id="page-12-20"></span>Liu, T. Y., Huang, T. K., Tseng, C. Y., Lai, Y. S., Lin, S. I., Lin, W. Y., et al. (2012). PHO2-dependent degradation of PHO1 modulates phosphate homeostasis in *Arabidopsis. Plant Cell* 24 (5), 2168–2183. doi: [10.1105/tpc.112.096636](https://doi.org/10.1105/tpc.112.096636)
- <span id="page-12-18"></span>Liu, W., Li, Q., Wang, Y., Wu, T., Yang, Y., Zhang, X., et al. (2017). Ethylene response factor AtERF72 negatively regulates *Arabidopsis thaliana* response to iron deficiency. *Biochem. Biophys. Res. Commun.* 491, 862–868. doi: [10.1016/j.](https://doi.org/10.1016/j.bbrc.2017.04.014) [bbrc.2017.04.014](https://doi.org/10.1016/j.bbrc.2017.04.014)
- <span id="page-12-16"></span>Lucena, C., Waters, B. M., Romera, F. J., García, M. J., Morales, M., Alcántara, E., et al. (2006). Ethylene could influence ferric reductase, iron transporter and H+-ATPase gene expression by affecting FER (or FER-like) gene activity. *J. Exp. Bot.* 57, 4145–4154. doi: [10.1093/jxb/erl189](https://doi.org/10.1093/jxb/erl189)
- <span id="page-12-24"></span>Lucena, C., Romera, F. J., Rojas, C. L., García, M. J., Alcántara, E., and Pérez-Vicente, R. (2007). Bicarbonate blocks the expression of several genes involved in the physiological responses to Fe deficiency of Strategy I plants. *Funct. Plant Biol.* 34, 1002–1009. doi: [10.1071/FP07136](https://doi.org/10.1071/FP07136)
- <span id="page-12-1"></span>Lucena, C., Romera, F. J., García, M. J., Alcántara, A., and Pérez-Vicente, R. (2015). Ethylene participates in the regulation of Fe deficiency responses in Strategy I plants and in rice. *Front. Plant Sci.* 6, 1056. doi: [10.3389/fpls.2015.01056](https://doi.org/10.3389/fpls.2015.01056)
- <span id="page-12-5"></span>Lucena, C., Porras, R., Romena, F. J., Alcántara, E., and Pérez-Vicente, R. (2018). Similarities and differences in the acquisition of Fe and P by dicot plants. *Agronomy* 8, 148–163. doi: [10.3390/agronomy8080148](https://doi.org/10.3390/agronomy8080148)
- <span id="page-12-0"></span>Marschner, H. (1995). *Mineral nutrition of higher plants*. 2nd ed edition. London: Academic Press.
- <span id="page-12-30"></span>Meng, Z. B., Chen, L. Q., Suo, D., Li, G. X., Tang, C. X., and Zheng, S. J. (2012). Nitric oxide is the shared signalling molecule in phosphorus- and iron deficiency- induced formation of cluster roots in white lupin (*Lupinus albus*). *Ann. Bot.* 109, 1055–1064. doi: [10.1093/aob/mcs024](https://doi.org/10.1093/aob/mcs024)
- <span id="page-12-9"></span>Mehra, P., Pandey, B. K., and Giri, J. (2017). Improvement in phosphate acquisition and utilization by a secretory purple acid phosphatase (OsPAP21b) in rice. *Plant Biotech. J.* 15 (8), 1054–1067. doi: [10.1111/pbi.12699](https://doi.org/10.1111/pbi.12699)
- <span id="page-12-12"></span>Misson, J., Raghothama, K. G., Jain, A., Jouhet, J., Block, M. A., Bligny, R., et al. (2005). A genome-wide transcriptional analysis using Arabidopsis thaliana

Affymetrix gene chips determined plant responses to phosphate deprivation. *Proc. Natl. Acad. Sci. Biol.* 102, 11934–11939. doi: [10.1073/pnas.0505266102](https://doi.org/10.1073/pnas.0505266102)

- <span id="page-12-31"></span>Miura, K., Lee, J., Gong, Q., Ma, S., Jin, J. B., Yoo, C. Y., et al. (2011). SIZ1 regulation of phosphate starvation induced root architecture remodeling involves the control of auxin accumulation. *Plant Physiol.* 155, 1000–1012. doi: [10.1104/](https://doi.org/10.1104/pp.110.165191) [pp.110.165191](https://doi.org/10.1104/pp.110.165191)
- <span id="page-12-25"></span>Mora, V., Baigorri, R., Bacaicoa, E., Zamarreño, A. M., and García-Mina, J. M. (2012). The humic acid-induced changes in the root concentration of nitric oxide, IAA and ethylene do not explain the changes in root architecture caused by humic acid in cucumber. *Environ. Exp. Bot.* 76, 24–32. doi: [10.1016/j.envexpbot.2011.10.001](https://doi.org/10.1016/j.envexpbot.2011.10.001)
- Müller, R., Morant, M., Jarmer, H., Nilsson, L., and Nielsen, T. H. (2007). Genome wide analysis of the *Arabidopsis* leaf transcriptome reveals interaction of phosphate and sugar metabolism. *Plant Physiol.* 143, 156–171. doi: [10.1104/](https://doi.org/10.1104/pp.106.090167) [pp.106.090167](https://doi.org/10.1104/pp.106.090167)
- <span id="page-12-26"></span>Nagarajan, V. K., and Smith, A. P. (2012). Ethylene´s role in phosphate starvation: more than just a root growth regulator. *Plant Cell Physiol.* 53 (2), 277–286. doi: [10.1093/pcp/pcr186](https://doi.org/10.1093/pcp/pcr186)
- <span id="page-12-3"></span>Neumann, G. (2016). The role of ethylene in plant adaptations for phosphate acquisition in soils—A review. *Front. Plant Sci.* 6, 1224. doi: [10.3389/](https://doi.org/10.3389/fpls.2015.01224) [fpls.2015.01224](https://doi.org/10.3389/fpls.2015.01224)
- <span id="page-12-22"></span>Pant, B. D., Musialak-Lange, M., Nuc, P., May, P., Buhtz, A., Kehr, J. et al. (2009). Identification of nutrient-responsive *Arabidopsis* and rapeseed microARNs by comprehensive real-time polymerase chain reaction profiling and small RNA sequencing. *Plant Physiol.* 150, 1541–1555. doi: [10.1104/pp.109.139139](https://doi.org/10.1104/pp.109.139139)
- Perea-García, A., García-Molina, A., Andres-Colas, N., Vera-Sirera, F., Perez-Amador, M. A., Puig, S., et al. (2013). *Arabidopsis* copper transport protein COPT2 participates in the cross talk between iron deficiency responses and low-phosphate. *Plant Physiol.* 162, 180–194. doi: [10.1104/pp.112.212407](https://doi.org/10.1104/pp.112.212407)
- <span id="page-12-8"></span>Robinson, W. D., Park, J., Tran, H. T., Del Vecchio, H. A., Ying, S., Zins, J. L., et al. (2012). The secreted purple acid phosphatase isozymes AtPAP12 and AtPAP26 play a pivotal role in extracellular phosphate-scavenging by *Arabidopsis thaliana. J. Exp. Bot.* 63 (18), 6531–6542. doi: [10.1093/jxb/ers309](https://doi.org/10.1093/jxb/ers309)
- <span id="page-12-14"></span>Romera, F. J., Alcantara, E., and de la Guardia, M. D. (1992). Effects of bicarbonate, phosphate and high pH on the reducing capacity of Fe-deficient sunflower and cucumber plants. *J. Plant Nutr.* 15, 1519–1530. doi: [10.1080/01904169209364418](https://doi.org/10.1080/01904169209364418)
- Romera, F. J., and Alcántara, E. (1994). Iron-deficiency stress responses in cucumber (*Cucumis sativus* L.) roots. A possible role for ethylene? *Plant Physiol.* 105, 1133–1138. doi: [10.1104/pp.105.4.1133](https://doi.org/10.1104/pp.105.4.1133)
- Roldán, M., Dinh, P., Leung, S., and McManus, M. T. (2013). Ethylene and the responses of plants to phosphate deficiency. *AoB PLANTS* 5, plt013. doi: [10.1093/aobpla/plt013](https://doi.org/10.1093/aobpla/plt013)
- <span id="page-12-10"></span>Rubio, V., Linhares, F., Solano, R., Martín, A. C., Iglesias, J., Leyva, A. et al. (2001). A conversed MYB transcription factor involved in phosphate starvation both in vascular plants and in unicellular algae. *Genes Dev.* 15, 2122–2133. doi: [10.1101/gad.204401](https://doi.org/10.1101/gad.204401)
- Sánchez-Rodriguez, A. R., del Campillo, M. C., and Torrent, J. (2013). Phosphate aggravates iron chlorosis in carbonate–iron oxide systems. *Plant Soil* 373, 31–42. doi: [10.1007/s11104-013-1785-y](https://doi.org/10.1007/s11104-013-1785-y)
- <span id="page-12-19"></span>Stacey, M. G., Patel, A., McClain, W. E., Mathieu, M., Remley, M., Rogers, E. E., et al. (2008). The *Arabidopsis* AtOPT3 protein functions in metal homeostasis and movement of iron to developing seeds. *Plant Physiol.* 146, 589–601. doi: [10.1104/pp.107.108183](https://doi.org/10.1104/pp.107.108183)
- <span id="page-12-4"></span>Stetter, M. G., Benz, M., and Ludewig, U. (2017). Increased root hair density by loss of WRKY6 in *Arabidopsis thaliana. Peer J.* 5, e2891. doi: [10.7717/peerj.2891](https://doi.org/10.7717/peerj.2891)
- <span id="page-12-2"></span>Song, L., and Liu, D. (2015). Ethylene and plant responses to phosphate deficiency. *Front. Plant Sci.* 6, 796. doi: [10.3389/fpls.2015.00796](https://doi.org/10.3389/fpls.2015.00796)
- <span id="page-12-28"></span>Thimm, O., Essigmann, B., Kloska, S., Altmann, T., and Buckhout, T. J. (2001). Response of *Arabidopsis* to iron deficiency stress as revealed by microarray analysis. *Plant Physiol.* 127 (3), 1030–1043. doi: [10.1104/pp.010191](https://doi.org/10.1104/pp.010191)
- <span id="page-12-11"></span>Todd, C. D., Zeng, P., Rodríguez, A. M., Hoyos., M. E., and Polacco, J. C. (2004). Transcripts of MYB-like genes respond to phosphorus and nitrogen deprivation in *Arabidopsis. Planta* 219, 1003–1009. doi: [10.1007/s00425-004-1305-7](https://doi.org/10.1007/s00425-004-1305-7)
- <span id="page-12-6"></span>Walker, E. L., and Connolly, E. L. (2008). Time to pump iron: iron-deficiency mechanisms of higher plants. *Curr. Opin. Plant Biol.* 11, 530–535. doi: [10.1016/j.pbi.2008.06.013](https://doi.org/10.1016/j.pbi.2008.06.013)
- <span id="page-12-27"></span>Wang, Y. H., Garvin, D. F., and Kochian, L. V. (2002). Rapid introduction of regulatory and transporter genes in response to phosphorus, potassium, and

{13}------------------------------------------------

iron deficiencies in tomato roots. Evidence for cross talk and root/rhizospheremediated signals. *Plant Physiol.* 130, 1370. doi: [10.1104/pp.008854](https://doi.org/10.1104/pp.008854)

- Wang, K. L. C., Li, H., and Ecker, J. R. (2002). Ethylene biosynthesis and networks. *Plant Cell* 2002, S131–S151. doi: [10.1105/tpc.001768](https://doi.org/10.1105/tpc.001768)
- <span id="page-13-2"></span>Wang, X., Wang, Y., Tian, J., Lim, B. L., Yan, X., and Liao, H. (2009). Overexpressing AtPAP15 enhances phosphorus efficiency in soybean. *Plant Physiol.* 151, 233– 240. doi: [10.1104/pp.109.138891](https://doi.org/10.1104/pp.109.138891)
- <span id="page-13-5"></span>Wang, Z., Straub, D., Yang, H., Kania, A., Shen, J., Ludewig, U., et al. (2014). The regulatory network of cluster-root function and development in phosphate deficient white lupin (*Lupinus albus*) identified by transcriptome sequencing. *Physiol. Plant* 151, 323–338. doi: [10.1111/ppl.12187](https://doi.org/10.1111/ppl.12187)
- <span id="page-13-7"></span>Wang, F., Deng, M., Xu, J., Zhu, X., and Mao, C. (2018). Molecular mechanisms of phosphate transport and in higher plants. *Semin. Cell Develop. Biol.* 74, 114– 122. doi: [10.1016/j.semcdb.2017.06.013](https://doi.org/10.1016/j.semcdb.2017.06.013)
- <span id="page-13-8"></span>Wang, L., and Liu, D. (2018). Functions and regulation of phosphate starvationinduced secreted acid phosphatases in higher plants. *Plant Sci.* 271, 108–116. doi: [10.1016/j.plantsci.2018.03.013](https://doi.org/10.1016/j.plantsci.2018.03.013)
- <span id="page-13-3"></span>Ward, J. T., Lahner, B., Yakubova, E., Salt, D. E., and Raghothama, K. G. (2008). The effect of iron on the primary root elongation of *Arabidopsis* during phosphate deficiency. *Plant Physiol.* 147, 1181–1191. doi: [10.1104/](https://doi.org/10.1104/pp.108.118562) [pp.108.118562](https://doi.org/10.1104/pp.108.118562)
- <span id="page-13-0"></span>Yuan, Y., Wu, H., Wang, N., Li, J., Zhao, W., Du, J., et al. (2008). FIT interacts with AtbHLH38 and AtbHLH39 in regulating iron uptake gene expression for iron homeostasis in *Arabidopsis. Cell Res.* 18, 385–397. doi: [10.1038/](https://doi.org/10.1038/cr.2008.26) [cr.2008.26](https://doi.org/10.1038/cr.2008.26)
- <span id="page-13-9"></span>Zakhleniuk, O. V., Raines, C. A., and Lloyd, J. C. (2001). Pho3: a phosphorusdeficient mutant of *Arabidopsis thaliana* (L.) Heynh. *Planta* 212, 529–534. doi: [10.1007/s004250000450](https://doi.org/10.1007/s004250000450)
- <span id="page-13-6"></span>Zhai, Z., Gayomba, S. R., Jung, H., Vimalakumari, N. K., Piñeros, M., Craft, E., et al. (2014). OPT3 is a Phloem-specific iron transporter that is essential for systemic iron and redistriburtion of iron and cadmium in *Arabidopsis. Plant Cell* 26, 2249–2264. doi: [10.1105/tpc.114.123737](https://doi.org/10.1105/tpc.114.123737)
- <span id="page-13-1"></span>Zhang, Z., Hong, L., and William, J. L. (2014). Molecular mechanisms underlying phosphate sensing, and adaptation in plants. *J. Integrative Plant Biol.* 56, 192– 220. doi: [10.1111/jipb.12163](https://doi.org/10.1111/jipb.12163)
- <span id="page-13-4"></span>Zheng, L., Huang, F., Narsai, R., Wu, J., Giraud, E., He, F., et al. (2009). Physiological and transcriptome analysis of iron and phosphorus interaction in rice seedlings. *Plant Physiol.* 151, 262–274. doi: [10.1104/pp.109.141051](https://doi.org/10.1104/pp.109.141051)

**Conflict of Interest:** The authors declare that the research was conducted in the absence of any commercial or financial relationships that could be construed as a potential conflict of interest.

*Copyright © 2019 Lucena, Porras, García, Alcántara, Pérez-Vicente, Zamarreño, Bacaicoa, García-Mina, Smith and Romera. This is an open-access article distributed under the terms of the [Creative Commons Attribution License \(CC BY\).](http://creativecommons.org/licenses/by/4.0/) The use, distribution or reproduction in other forums is permitted, provided the original author(s) and the copyright owner(s) are credited and that the original publication in this journal is cited, in accordance with accepted academic practice. No use, distribution or reproduction is permitted which does not comply with these terms.*""",
]

# Sample reference strings from CEX and EXCITE benchmarks for parsing
SAMPLE_REFERENCES = [
    # From CEX - AGR-BIO-SCI dairy farm study
    [
        "B Algers, G Bertoni, D Broom, J Hartung, L Lidfors, J Metz, L Munksgaard, T N Pina, P Oltenacu, J Rehage, J Rushen. Scientific report on the effects of farming systems on dairy cow welfare and disease. Annex to the EFSA Journal. 2009. Vol. 1143",
        "E Burow, T Rousing, P Thomsen, D Otten, J Sørensen. Effect of grazing on the cow welfare of dairy herds evaluated by a multidimensional welfare index.. Animal. 2013a. Vol. 7",
        "D Gieseke, C Lambertz, M Gauly. Relationship between herd size and animal welfare in dairy cattle. Journal of Dairy Science. 2018. Vol. 101",
    ],
    # From EXCITE - German sociology paper
    [
        "Arends-Tòth, J. / Van de Vijver, F. J. (2003): Multiculturalism and acculturation: Views of Dutch and Turkish-Dutch. European Journal of Social Psychology 33(2), S. 249-266.",
        "Berry, J. (1997): Immigration, acculturation and adaption. Applied Psychology 46(1), S. 5-34.",
        "Phinney, J. S. (1992): The Multigroup Ethnic Identity Measure: A new scale for use with diverse groups. Journal of Adolescent Research 7(2), S. 156- 176.",
    ],
    # From EXCITE - German census data study
    [
        "Statistisches Bundesamt (1978a): Volkszählung vom 27. Mai 1970. Methodische und praktische Vorbereitung sowie Durchführung der Volkszählung 1970. Fachserie 1 Heft 25. Stuttgart/Mainz: Kohlhammer.",
        "Bach, Walter (1979): Ziehung einer Stichprobe aus den Mikrodaten der Berufszählung 1970. Mannheim: VASMA-Arbeitspapier 9.",
        "Schimpl-Neimanns, Bernhard/Frenzel, Hansjörg (1995): 1-Prozent Stichprobe der Volks- und Berufszählung 1970 – Datei mit Haushalts- und Familiennummern und revidierter Teilstichprobe für West-Berlin. Dokumentation der Datenaufbereitung. Mannheim: ZUMA-Technischer Bericht T95/06.",
    ],
]


# Get workspace root
WORKSPACE_ROOT = Path(__file__).parent

# Sample PDFs from benchmarks
SAMPLE_PDFS = [
    # CEX benchmark PDFs (diverse domains)
    WORKSPACE_ROOT / "benchmarks/cex/all_pdfs/COM-SCI_25.pdf",
    WORKSPACE_ROOT / "benchmarks/cex/all_pdfs/PSY_97.pdf",
    WORKSPACE_ROOT / "benchmarks/cex/all_pdfs/MATH_69.pdf",
    # EXCITE benchmark PDFs
    WORKSPACE_ROOT / "benchmarks/excite/all_pdfs/1219.pdf",
    WORKSPACE_ROOT / "benchmarks/excite/all_pdfs/4605.pdf",
    WORKSPACE_ROOT / "benchmarks/excite/all_pdfs/22622.pdf",
]


def submit_text_extraction_job(pdf_path: Path, job_num: int) -> Dict:
    """Submit a text extraction job (PDF -> text)."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    print(f"[TextExtraction {job_num}] Submitting {pdf_path.name}...")
    
    with open(pdf_path, 'rb') as f:
        files = {'file': (pdf_path.name, f, 'application/pdf')}
        params = {'extractor': 'pymupdf', 'markdown': True}
        
        response = requests.post(
            f"{API_BASE}/extract/text",
            files=files,
            params=params
        )
        response.raise_for_status()
    
    result = response.json()
    job_id = result["job_id"]
    print(f"[TextExtraction {job_num}] Job ID: {job_id} [default queue]")
    
    return {
        "type": "text_extraction",
        "job_num": job_num,
        "job_id": job_id,
        "pdf_name": pdf_path.name,
        "submitted_at": time.time(),
    }


def submit_extraction_job(text: str, job_num: int) -> Dict:
    """Submit a reference extraction job."""
    print(f"[Extraction {job_num}] Submitting...")
    
    response = requests.post(
        f"{API_BASE}/extract/references",
        json={"text": text},
        params={"method": "full_text", "temperature": 0.3}
    )
    response.raise_for_status()
    
    result = response.json()
    job_id = result["job_id"]
    print(f"[Extraction {job_num}] Job ID: {job_id} [llm-tasks queue]")
    
    return {
        "type": "extraction",
        "job_num": job_num,
        "job_id": job_id,
        "submitted_at": time.time(),
    }


def submit_parsing_job(references: List[str], job_num: int) -> Dict:
    """Submit a reference parsing job."""
    print(f"[Parsing {job_num}] Submitting {len(references)} references...")
    
    response = requests.post(
        f"{API_BASE}/parse/references",
        json={"references": references},
        params={"parser": "llm", "temperature": 0.0}
    )
    response.raise_for_status()
    
    result = response.json()
    job_id = result["job_id"]
    print(f"[Parsing {job_num}] Job ID: {job_id} [llm-tasks queue]")
    
    return {
        "type": "parsing",
        "job_num": job_num,
        "job_id": job_id,
        "submitted_at": time.time(),
    }


def poll_job(job_info: Dict, max_wait: int = 300) -> Dict:
    """Poll a job until completion or timeout."""
    job_id = job_info["job_id"]
    job_type = job_info["type"]
    job_num = job_info["job_num"]
    
    start_time = time.time()
    last_status = None
    
    while (time.time() - start_time) < max_wait:
        try:
            response = requests.get(f"{API_BASE}/jobs/{job_id}/status")
            response.raise_for_status()
            status_data = response.json()
            
            current_status = status_data["status"]
            
            # Print status changes
            if current_status != last_status:
                elapsed = time.time() - job_info["submitted_at"]
                print(f"[{job_type.capitalize()} {job_num}] Status: {current_status} (after {elapsed:.1f}s)")
                last_status = current_status
            
            if current_status == "completed":
                # Fetch result
                result_response = requests.get(f"{API_BASE}/jobs/{job_id}")
                result_response.raise_for_status()
                result = result_response.json()
                
                total_time = time.time() - job_info["submitted_at"]
                
                if job_type == "text_extraction":
                    text_len = len(result.get("text", ""))
                    print(f"[{job_type.capitalize()} {job_num}] ✓ Completed in {total_time:.1f}s - {text_len} chars extracted")
                elif job_type == "extraction":
                    ref_count = result.get("count", len(result.get("references", [])))
                    print(f"[{job_type.capitalize()} {job_num}] ✓ Completed in {total_time:.1f}s - {ref_count} references extracted")
                else:
                    ref_count = result.get("count", len(result.get("references", [])))
                    print(f"[{job_type.capitalize()} {job_num}] ✓ Completed in {total_time:.1f}s - {ref_count} references parsed")
                
                return {
                    **job_info,
                    "status": "completed",
                    "total_time": total_time,
                    "result": result,
                }
            
            elif current_status == "failed":
                error = status_data.get("error", "Unknown error")
                print(f"[{job_type.capitalize()} {job_num}] ✗ Failed: {error}")
                return {
                    **job_info,
                    "status": "failed",
                    "error": error,
                }
            
            # Wait before polling again
            time.sleep(2)
            
        except requests.RequestException as e:
            print(f"[{job_type.capitalize()} {job_num}] Error polling: {e}")
            time.sleep(2)
    
    print(f"[{job_type.capitalize()} {job_num}] ⏱ Timeout after {max_wait}s")
    return {
        **job_info,
        "status": "timeout",
    }


def test_concurrent_text_extractions(num_jobs: int = 3):
    """Test multiple concurrent text extraction jobs (PDF -> text)."""
    print("\n" + "="*70)
    print(f"TEST 1: {num_jobs} Concurrent Text Extraction Jobs (default queue)")
    print("="*70 + "\n")
    
    start_time = time.time()
    
    # Check which PDFs exist
    available_pdfs = [pdf for pdf in SAMPLE_PDFS if pdf.exists()]
    if not available_pdfs:
        print("⚠ No sample PDFs found in benchmarks/ - skipping text extraction test")
        return []
    
    # Use available PDFs (cycle if needed)
    pdfs_to_test = [available_pdfs[i % len(available_pdfs)] for i in range(num_jobs)]
    
    # Submit all jobs
    jobs = []
    with ThreadPoolExecutor(max_workers=num_jobs) as executor:
        futures = [
            executor.submit(submit_text_extraction_job, pdfs_to_test[i], i+1)
            for i in range(num_jobs)
        ]
        
        for future in as_completed(futures):
            try:
                jobs.append(future.result())
            except Exception as e:
                print(f"✗ Failed to submit job: {e}")
    
    print(f"\n✓ All {num_jobs} jobs submitted\n")
    
    # Poll all jobs (text extraction timeout)
    results = []
    with ThreadPoolExecutor(max_workers=num_jobs) as executor:
        futures = [executor.submit(poll_job, job, TIMEOUT_TEXT_EXTRACTION) for job in jobs]
        
        for future in as_completed(futures):
            results.append(future.result())
    
    # Summary
    total_time = time.time() - start_time
    completed = sum(1 for r in results if r["status"] == "completed")
    failed = sum(1 for r in results if r["status"] == "failed")
    
    print(f"\n{'='*70}")
    print(f"TEST 1 SUMMARY:")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Completed: {completed}/{num_jobs}")
    print(f"  Failed: {failed}/{num_jobs}")
    print(f"{'='*70}\n")
    
    return results


def test_concurrent_extractions(num_jobs: int = 3):
    """Test multiple concurrent reference extraction jobs (text -> references)."""
    print("\n" + "="*70)
    print(f"TEST 2: {num_jobs} Concurrent Reference Extraction Jobs (llm-tasks queue)")
    print("="*70 + "\n")
    
    start_time = time.time()
    
    # Submit all jobs
    jobs = []
    with ThreadPoolExecutor(max_workers=num_jobs) as executor:
        futures = [
            executor.submit(submit_extraction_job, SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)], i+1)
            for i in range(num_jobs)
        ]
        
        for future in as_completed(futures):
            jobs.append(future.result())
    
    print(f"\n✓ All {num_jobs} jobs submitted\n")
    
    # Poll all jobs (reference extraction timeout)
    results = []
    with ThreadPoolExecutor(max_workers=num_jobs) as executor:
        futures = [executor.submit(poll_job, job, TIMEOUT_REFERENCE_EXTRACTION) for job in jobs]
        
        for future in as_completed(futures):
            results.append(future.result())
    
    # Summary
    total_time = time.time() - start_time
    completed = sum(1 for r in results if r["status"] == "completed")
    failed = sum(1 for r in results if r["status"] == "failed")
    
    print(f"\n{'='*70}")
    print(f"TEST 2 SUMMARY:")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Completed: {completed}/{num_jobs}")
    print(f"  Failed: {failed}/{num_jobs}")
    print(f"{'='*70}\n")
    
    return results


def test_concurrent_parsing(num_jobs: int = 3):
    """Test multiple concurrent reference parsing jobs (references -> structured)."""
    print("\n" + "="*70)
    print(f"TEST 3: {num_jobs} Concurrent Reference Parsing Jobs (llm-tasks queue)")
    print("="*70 + "\n")
    
    start_time = time.time()
    
    # Submit all jobs
    jobs = []
    with ThreadPoolExecutor(max_workers=num_jobs) as executor:
        futures = [
            executor.submit(submit_parsing_job, SAMPLE_REFERENCES[i % len(SAMPLE_REFERENCES)], i+1)
            for i in range(num_jobs)
        ]
        
        for future in as_completed(futures):
            jobs.append(future.result())
    
    print(f"\n✓ All {num_jobs} jobs submitted\n")
    
    # Poll all jobs (reference parsing timeout)
    results = []
    with ThreadPoolExecutor(max_workers=num_jobs) as executor:
        futures = [executor.submit(poll_job, job, TIMEOUT_REFERENCE_PARSING) for job in jobs]
        
        for future in as_completed(futures):
            results.append(future.result())
    
    # Summary
    total_time = time.time() - start_time
    completed = sum(1 for r in results if r["status"] == "completed")
    failed = sum(1 for r in results if r["status"] == "failed")
    
    print(f"\n{'='*70}")
    print(f"TEST 3 SUMMARY:")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Completed: {completed}/{num_jobs}")
    print(f"  Failed: {failed}/{num_jobs}")
    print(f"{'='*70}\n")
    
    return results


def test_mixed_workload(num_text_extraction: int = 2, num_extraction: int = 2, num_parsing: int = 2):
    """Test mixed workload with all three job types across both queues."""
    print("\n" + "="*70)
    print(f"TEST 4: Mixed Workload ({num_text_extraction} text extraction + {num_extraction} ref extraction + {num_parsing} parsing)")
    print("="*70 + "\n")
    
    start_time = time.time()
    
    # Check which PDFs exist
    available_pdfs = [pdf for pdf in SAMPLE_PDFS if pdf.exists()]
    if not available_pdfs and num_text_extraction > 0:
        print("⚠ No sample PDFs found - skipping text extraction in mixed test")
        num_text_extraction = 0
    
    # Use available PDFs (cycle if needed)
    pdfs_to_test = [available_pdfs[i % len(available_pdfs)] for i in range(num_text_extraction)] if available_pdfs else []
    
    # Submit all jobs concurrently
    jobs = []
    total_jobs = num_text_extraction + num_extraction + num_parsing
    with ThreadPoolExecutor(max_workers=total_jobs) as executor:
        futures = []
        
        # Submit text extraction jobs (default queue)
        for i in range(num_text_extraction):
            futures.append(
                executor.submit(submit_text_extraction_job, pdfs_to_test[i], i+1)
            )
        
        # Submit reference extraction jobs (llm-tasks queue)
        for i in range(num_extraction):
            futures.append(
                executor.submit(submit_extraction_job, SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)], i+1)
            )
        
        # Submit parsing jobs (llm-tasks queue)
        for i in range(num_parsing):
            futures.append(
                executor.submit(submit_parsing_job, SAMPLE_REFERENCES[i % len(SAMPLE_REFERENCES)], i+1)
            )
        
        for future in as_completed(futures):
            try:
                jobs.append(future.result())
            except Exception as e:
                print(f"✗ Failed to submit job: {e}")
    
    print(f"\n✓ All {total_jobs} jobs submitted\n")
    
    # Poll all jobs (use max timeout since we have mixed job types)
    results = []
    max_timeout = max(TIMEOUT_REFERENCE_EXTRACTION, TIMEOUT_REFERENCE_PARSING)
    with ThreadPoolExecutor(max_workers=num_extraction + num_parsing) as executor:
        futures = [executor.submit(poll_job, job, max_timeout) for job in jobs]
        
        for future in as_completed(futures):
            results.append(future.result())
    
    # Summary
    total_time = time.time() - start_time
    completed = sum(1 for r in results if r["status"] == "completed")
    failed = sum(1 for r in results if r["status"] == "failed")
    text_extraction_completed = sum(1 for r in results if r["status"] == "completed" and r["type"] == "text_extraction")
    extraction_completed = sum(1 for r in results if r["status"] == "completed" and r["type"] == "extraction")
    parsing_completed = sum(1 for r in results if r["status"] == "completed" and r["type"] == "parsing")
    
    print(f"\n{'='*70}")
    print(f"TEST 4 SUMMARY:")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Completed: {completed}/{total_jobs}")
    print(f"    - Text Extraction (default queue): {text_extraction_completed}/{num_text_extraction}")
    print(f"    - Ref Extraction (llm-tasks queue): {extraction_completed}/{num_extraction}")
    print(f"    - Parsing (llm-tasks queue): {parsing_completed}/{num_parsing}")
    print(f"  Failed: {failed}/{total_jobs}")
    print(f"{'='*70}\n")
    
    return results


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("CONCURRENT QUEUE TEST SUITE")
    print("="*70)
    print(f"API: {API_BASE}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Check API health
    try:
        response = requests.get(f"{API_BASE}/health")
        response.raise_for_status()
        print("✓ API is healthy\n")
    except requests.RequestException as e:
        print(f"✗ API health check failed: {e}")
        print("Make sure the API server is running: python -m citation_index.api")
        return
    
    # Run tests
    results = {}
    
    try:
        # Test 1: Concurrent text extractions (default queue)
        results["text_extractions"] = test_concurrent_text_extractions(num_jobs=3)
        
        # Test 2: Concurrent reference extractions (llm-tasks queue)
        results["extractions"] = test_concurrent_extractions(num_jobs=3)
        
        # Test 3: Concurrent parsing (llm-tasks queue)
        results["parsing"] = test_concurrent_parsing(num_jobs=3)
        
        # Test 4: Mixed workload (both queues)
        results["mixed"] = test_mixed_workload(num_text_extraction=2, num_extraction=2, num_parsing=2)
        
        # Final summary
        print("\n" + "="*70)
        print("FINAL SUMMARY")
        print("="*70)
        
        total_jobs = sum(len(results[k]) for k in results)
        total_completed = sum(
            sum(1 for r in results[k] if r["status"] == "completed")
            for k in results
        )
        total_failed = sum(
            sum(1 for r in results[k] if r["status"] == "failed")
            for k in results
        )
        
        print(f"Total jobs: {total_jobs}")
        print(f"Completed: {total_completed}")
        print(f"Failed: {total_failed}")
        print(f"Success rate: {total_completed/total_jobs*100:.1f}%")
        print("="*70 + "\n")
        
        # Save results to file
        output_file = Path("test_queue_results.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"✓ Results saved to {output_file}")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Test interrupted by user")
    except Exception as e:
        print(f"\n\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
