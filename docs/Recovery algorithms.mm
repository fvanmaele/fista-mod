<map version="freeplane 1.9.13">
<!--To view this file, download free mind mapping software Freeplane from https://www.freeplane.org -->
<node TEXT="Recovery algorithms" LOCALIZED_STYLE_REF="AutomaticLayout.level.root" FOLDED="false" ID="Freemind_Link_1513112588" CREATED="1153430895318" MODIFIED="1666366811537" STYLE="bubble" SHAPE_HORIZONTAL_MARGIN="10 pt" SHAPE_VERTICAL_MARGIN="10 pt">
<hook NAME="accessories/plugins/AutomaticLayout.properties" VALUE="ALL"/>
<hook NAME="MapStyle" background="#f5f5dc">
    <properties show_icon_for_attributes="true" edgeColorConfiguration="#808080ff,#ff0000ff,#0000ffff,#00ff00ff,#ff00ffff,#00ffffff,#7c0000ff,#00007cff,#007c00ff,#7c007cff,#007c7cff,#7c7c00ff" show_note_icons="true" associatedTemplateLocation="template:/light_sepia_template.mm" fit_to_viewport="false"/>

<map_styles>
<stylenode LOCALIZED_TEXT="styles.root_node" STYLE="oval" UNIFORM_SHAPE="true" VGAP_QUANTITY="24 pt">
<font SIZE="24"/>
<stylenode LOCALIZED_TEXT="styles.predefined" POSITION="right" STYLE="bubble">
<stylenode LOCALIZED_TEXT="default" ID="ID_1558627382" ICON_SIZE="12 pt" FORMAT_AS_HYPERLINK="false" COLOR="#2c2b29" BACKGROUND_COLOR="#eedfcc" STYLE="bubble" SHAPE_HORIZONTAL_MARGIN="8 pt" SHAPE_VERTICAL_MARGIN="5 pt" BORDER_WIDTH_LIKE_EDGE="false" BORDER_WIDTH="1.9 px" BORDER_COLOR_LIKE_EDGE="true" BORDER_COLOR="#f0f0f0" BORDER_DASH_LIKE_EDGE="true" BORDER_DASH="SOLID" VGAP_QUANTITY="3 pt">
<arrowlink SHAPE="CUBIC_CURVE" COLOR="#000000" WIDTH="2" TRANSPARENCY="200" DASH="" FONT_SIZE="9" FONT_FAMILY="SansSerif" DESTINATION="ID_1558627382" STARTINCLINATION="81.75 pt;-9.75 pt;" ENDINCLINATION="81.75 pt;19.5 pt;" STARTARROW="NONE" ENDARROW="DEFAULT"/>
<font NAME="SansSerif" SIZE="9" BOLD="false" STRIKETHROUGH="false" ITALIC="false"/>
<edge STYLE="bezier" COLOR="#2e3440" WIDTH="3" DASH="SOLID"/>
<richcontent CONTENT-TYPE="plain/auto" TYPE="DETAILS"/>
<richcontent TYPE="NOTE" CONTENT-TYPE="plain/auto"/>
</stylenode>
<stylenode LOCALIZED_TEXT="defaultstyle.details" COLOR="#ffffff" BACKGROUND_COLOR="#2e3440">
<font SIZE="10"/>
</stylenode>
<stylenode LOCALIZED_TEXT="defaultstyle.attributes">
<font SIZE="10"/>
</stylenode>
<stylenode LOCALIZED_TEXT="defaultstyle.note" COLOR="#000000" BACKGROUND_COLOR="#f6f9a1" TEXT_ALIGN="LEFT">
<icon BUILTIN="clock2"/>
<font SIZE="10" ITALIC="true"/>
</stylenode>
<stylenode LOCALIZED_TEXT="defaultstyle.floating">
<edge STYLE="hide_edge"/>
<cloud COLOR="#f0f0f0" SHAPE="ROUND_RECT"/>
</stylenode>
<stylenode LOCALIZED_TEXT="defaultstyle.selection" COLOR="#2c2b29" BACKGROUND_COLOR="#ffffff" BORDER_COLOR_LIKE_EDGE="false" BORDER_COLOR="#bf616a"/>
</stylenode>
<stylenode LOCALIZED_TEXT="styles.user-defined" POSITION="right" STYLE="bubble">
<stylenode LOCALIZED_TEXT="styles.important" ID="ID_411331447" COLOR="#ffffff" BACKGROUND_COLOR="#bf616a" BORDER_COLOR="#bf616a">
<icon BUILTIN="yes"/>
<arrowlink COLOR="#bf616a" TRANSPARENCY="255" DESTINATION="ID_411331447"/>
<font NAME="Ubuntu" SIZE="12" BOLD="true"/>
<edge COLOR="#bf616a"/>
</stylenode>
</stylenode>
<stylenode LOCALIZED_TEXT="styles.AutomaticLayout" POSITION="right" STYLE="bubble">
<stylenode LOCALIZED_TEXT="AutomaticLayout.level.root" COLOR="#ffffff" BACKGROUND_COLOR="#2c2b29" STYLE="bubble" SHAPE_HORIZONTAL_MARGIN="10 pt" SHAPE_VERTICAL_MARGIN="10 pt" BORDER_COLOR_LIKE_EDGE="false" BORDER_COLOR="#2c2b29" BORDER_DASH_LIKE_EDGE="true">
<font SIZE="18"/>
</stylenode>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,1" STYLE="bubble" SHAPE_HORIZONTAL_MARGIN="8 pt" SHAPE_VERTICAL_MARGIN="5 pt" BORDER_COLOR="#2c2b29">
<font SIZE="16"/>
</stylenode>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,2" STYLE="bubble" SHAPE_HORIZONTAL_MARGIN="8 pt" SHAPE_VERTICAL_MARGIN="5 pt" BORDER_COLOR_LIKE_EDGE="true" BORDER_COLOR="#f0f0f0">
<font SIZE="14"/>
</stylenode>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,3" STYLE="bubble" SHAPE_HORIZONTAL_MARGIN="8 pt" SHAPE_VERTICAL_MARGIN="5 pt" BORDER_COLOR_LIKE_EDGE="true" BORDER_COLOR="#f0f0f0">
<font SIZE="12"/>
</stylenode>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,4" BACKGROUND_COLOR="#eedfcc" BORDER_COLOR_LIKE_EDGE="true" BORDER_COLOR="#f0f0f0">
<font SIZE="11"/>
</stylenode>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,5" BORDER_COLOR_LIKE_EDGE="true" BORDER_COLOR="#f0f0f0">
<font SIZE="11"/>
</stylenode>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,6" BORDER_COLOR_LIKE_EDGE="true" BORDER_COLOR="#f0f0f0">
<font SIZE="10"/>
</stylenode>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,7" BORDER_COLOR="#f0f0f0">
<font SIZE="10"/>
</stylenode>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,8" BORDER_COLOR="#f0f0f0">
<font SIZE="10"/>
</stylenode>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,9" BORDER_COLOR="#f0f0f0">
<font SIZE="10"/>
</stylenode>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,10" BORDER_COLOR="#f0f0f0"/>
</stylenode>
</stylenode>
</map_styles>
</hook>
<node TEXT="Greedy pursuits" POSITION="right" ID="ID_1196073941" CREATED="1666366811929" MODIFIED="1666366814238">
<node TEXT="Matching pursuit" ID="ID_1180772102" CREATED="1666366824627" MODIFIED="1666366827296">
<node TEXT="Atomic selection step" ID="ID_205680454" CREATED="1666366836557" MODIFIED="1666366839204"/>
<node TEXT="Residual update step" ID="ID_1761710855" CREATED="1666366839341" MODIFIED="1666366841305"/>
</node>
<node TEXT="Orthogonal Matching Pursuit" ID="ID_819017370" CREATED="1666366827568" MODIFIED="1666366977687">
<hook URI="Algorithm_2_OMP.png" SIZE="0.6808871" NAME="ExternalObject"/>
<node TEXT="Exact recovery condition" ID="ID_1555608410" CREATED="1666366989272" MODIFIED="1666367097072">
<font ITALIC="true"/>
<node TEXT="\latex $\|A_S^+ A_{S^c} \|_{1} &lt; 1$" ID="ID_1739369290" CREATED="1666367006707" MODIFIED="1666367086830"/>
</node>
<node TEXT="RIP" ID="ID_24524139" CREATED="1666367779742" MODIFIED="1666367822213">
<font ITALIC="true"/>
<node TEXT="Order s+1" ID="ID_972277407" CREATED="1666367789771" MODIFIED="1666367820449"/>
<node TEXT="\latex $\delta_{s+1} &lt; 1/(3 \sqrt{s})" ID="ID_593621106" CREATED="1666367800865" MODIFIED="1666367816162"/>
</node>
<node TEXT="Normalized columns" ID="ID_1432892362" CREATED="1666367122272" MODIFIED="1666367124867"/>
</node>
<node TEXT="Matching pursuit" ID="ID_1041100224" CREATED="1666367113563" MODIFIED="1666367146885">
<hook URI="Algorithm_3_MP.png" SIZE="0.67944056" NAME="ExternalObject"/>
<node TEXT="Normalized columns" ID="ID_1501584430" CREATED="1666367126222" MODIFIED="1666367336801"/>
</node>
</node>
<node TEXT="Thresholding algorithms" POSITION="right" ID="ID_74489311" CREATED="1666366814429" MODIFIED="1666367510078" VGAP_QUANTITY="3 pt">
<node TEXT="Hard thresholding" ID="ID_1350122880" CREATED="1666367156479" MODIFIED="1666367477931">
<node TEXT="Solve ties" ID="ID_908800717" CREATED="1666367159628" MODIFIED="1666367161322">
<node TEXT="Lexicographic order" ID="ID_1439219303" CREATED="1666367161444" MODIFIED="1666367165072"/>
<node TEXT="Random choice" ID="ID_631566014" CREATED="1666367165222" MODIFIED="1666367166495"/>
</node>
<node TEXT="Basic thresholding" ID="ID_1274190659" CREATED="1666367185359" MODIFIED="1666367206047">
<hook URI="Algorithm_4_BT.png" SIZE="0.64863956" NAME="ExternalObject"/>
<node TEXT="Recovery condition" ID="ID_651560965" CREATED="1666367212607" MODIFIED="1666367215207">
<node TEXT="\latex $\min_{j \in S} |A^T b)_j| &gt; \max_{\ell \in S^c} |(A^T b)_\ell|" ID="ID_1157741846" CREATED="1666367215316" MODIFIED="1666367283410"/>
</node>
</node>
<node TEXT="Iterative hard thresholding" ID="ID_226899026" CREATED="1666367289628" MODIFIED="1666367417342">
<hook URI="Algorithm_5_IHT.png" SIZE="0.651434" NAME="ExternalObject"/>
<node TEXT="Step size" ID="ID_142943373" CREATED="1666369404962" MODIFIED="1666369427759">
<node TEXT="RIP" ID="ID_101760389" CREATED="1666369420461" MODIFIED="1666369429175">
<node TEXT="Random matrices" ID="ID_1017615277" CREATED="1666379035159" MODIFIED="1666379037502"/>
</node>
<node TEXT="Adaptive" ID="ID_629980319" CREATED="1666378723178" MODIFIED="1666378724656"/>
</node>
</node>
<node TEXT="Hard thresholding pursuit" ID="ID_388151233" CREATED="1666367438302" MODIFIED="1666367460317">
<hook URI="Algorithm_6_HTP.png" SIZE="0.6515656" NAME="ExternalObject"/>
</node>
</node>
</node>
<node TEXT="Greedy pursuits with thresholding" POSITION="left" ID="ID_993352340" CREATED="1666367613064" MODIFIED="1666367618934">
<node TEXT="Subspace pursuit" ID="ID_154078380" CREATED="1666367527124" MODIFIED="1667150328583">
<hook URI="Algorithm_8_SP.png" SIZE="0.56710774" NAME="ExternalObject"/>
<node TEXT="Normalized columns" ID="ID_446163063" CREATED="1666367537604" MODIFIED="1666367542896"/>
</node>
<node TEXT="CoSaMP" ID="ID_249565084" CREATED="1666367531968" MODIFIED="1666367691991">
<hook URI="Algorithm_7_CoSaMP.png" SIZE="0.64211" NAME="ExternalObject"/>
<node TEXT="Normalized columns" ID="ID_194254257" CREATED="1666367544210" MODIFIED="1666367546601"/>
</node>
</node>
</node>
</map>
